"""
Distributed training entry point using torchrun.

This script is designed to be launched with torchrun:
    torchrun --nproc_per_node=2 scripts/train_distributed.py

It sets up the distributed process group and demonstrates proper
multi-GPU training structure with:
- DistributedSampler for data partitioning
- Rank-aware logging
- Epoch synchronization across workers
"""

import argparse
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms

# Add project root to path
sys.path.insert(0, ".")

from src.coci.distributed import (
    setup_distributed,
    cleanup_distributed,
    get_rank,
    get_world_size,
    get_local_rank,
    is_main_process,
    is_distributed_initialized,
    barrier,
    log_on_main,
    reduce_metrics,
)
from src.coci.models.model import get_model
from src.coci.config import load_config
from src.coci.checkpointing.checkpoint_manager import CheckpointManager
from src.coci.metrics import MetricsCollector, MetricsExporter
from src.coci.fault.checkpoint_fault_injector import (
    CheckpointBitFlipInjector,
    CheckpointFaultConfig,
    FaultType,
    enforce_determinism,
)


# -------------------------------------------------
# Worker Seeding for Reproducibility
# -------------------------------------------------
def seed_worker(worker_id):
    """
    Seed worker for reproducible data loading across epochs.
    Ensures each worker uses a different but deterministic random seed.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


# -------------------------------------------------
# CIFAR-100 Dataset
# -------------------------------------------------
def get_cifar100_dataset(root="./data", train=True):
    """Get CIFAR-100 dataset with standard transforms."""
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761),
            ),
        ]
    )

    dataset = datasets.CIFAR100(
        root=root,
        train=train,
        download=True,
        transform=transform,
    )

    return dataset


# -------------------------------------------------
# Distributed DataLoader
# -------------------------------------------------
def get_distributed_dataloader(
    dataset, batch_size, num_workers, rank, world_size, shuffle=False
):
    """
    Create a DataLoader with DistributedSampler for proper data partitioning.

    Args:
        dataset: The dataset to load from.
        batch_size: Batch size per process.
        num_workers: Number of data loading workers.
        rank: Current process rank.
        world_size: Total number of processes.
        shuffle: Whether to shuffle data (handled by sampler, not DataLoader).

    Returns:
        DataLoader: Configured DataLoader with DistributedSampler.
    """
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=42,
        drop_last=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
    )

    return dataloader


# -------------------------------------------------
# Training Function
# -------------------------------------------------
def train_epoch(model, train_loader, optimizer, device, epoch, sampler):
    """
    Train for one epoch.

    Args:
        model: The neural network model.
        train_loader: Training data loader.
        optimizer: Optimizer.
        device: Device to train on.
        epoch: Current epoch number.
        sampler: DistributedSampler to synchronize epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Set epoch for DistributedSampler to ensure proper data shuffling
    # across epochs and across all processes
    if sampler is not None:
        sampler.set_epoch(epoch)

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


# -------------------------------------------------
# Evaluation Function
# -------------------------------------------------
def evaluate(model, test_loader, device):
    """
    Evaluate the model on test data.

    Args:
        model: The neural network model.
        test_loader: Test data loader.
        device: Device to evaluate on.

    Returns:
        tuple: (correct, total, accuracy) - counts and percentage for aggregation.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return correct, total, accuracy


# -------------------------------------------------
# Main Training
# -------------------------------------------------
def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Distributed Training")
    parser.add_argument("--mode", choices=["dev", "server"], default="dev")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--no-fault-injection",
        action="store_true",
        help="Disable checkpoint bit-flip injection (enabled by default)",
    )
    parser.add_argument(
        "--fault-type",
        type=str,
        default=FaultType.RANDOM_BIT.value,
        choices=[ft.value for ft in FaultType],
        help="Fault type for checkpoint bit-flip injection",
    )
    parser.add_argument(
        "--fault-location",
        type=str,
        default="model",
        help="Fault location: model, optimizer, all, or layer index",
    )
    parser.add_argument(
        "--fault-bit-range",
        type=str,
        default=None,
        help="Inclusive bit range low,high (example: 0,31)",
    )
    parser.add_argument(
        "--fault-probability",
        type=float,
        default=1.0,
        help="Fault probability per attempted injection",
    )
    parser.add_argument(
        "--fault-bit-flips",
        type=int,
        default=1,
        help="Number of bit flips to inject per resumed checkpoint",
    )
    parser.add_argument(
        "--fault-num-processes",
        type=int,
        default=1,
        help="How many ranks load a corrupted checkpoint",
    )
    parser.add_argument(
        "--fault-target-layers",
        type=str,
        default="",
        help="Comma-separated layer indices to target (model only)",
    )
    parser.add_argument(
        "--fault-specific-bit",
        type=int,
        default=None,
        help="Bit position for SPECIFIC_BIT fault type",
    )
    parser.add_argument(
        "--fault-log-path",
        type=str,
        default=None,
        help="Path to write injection logs",
    )
    parser.add_argument(
        "--fault-load-log",
        type=str,
        default=None,
        help="Path to an existing injection log to replay",
    )
    parser.add_argument(
        "--fault-seed",
        type=int,
        default=42,
        help="Deterministic seed for equivalent multi-rank injection",
    )
    args = parser.parse_args()

    # -------------------------
    # Distributed Setup
    # -------------------------
    rank, world_size, local_rank = setup_distributed(backend="nccl")

    enforce_determinism(args.fault_seed)
    random.seed(args.fault_seed + rank)
    np.random.seed(args.fault_seed + rank)

    # Set CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_distributed_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")

    # -------------------------
    # Configuration
    # -------------------------
    config_path = "configs/dev.yaml" if args.mode == "dev" else "configs/server.yaml"
    cfg = load_config(config_path)

    if args.epochs is not None:
        cfg.epochs = args.epochs

    # -------------------------
    # Logging (rank 0 only)
    # -------------------------
    log_on_main(f"\n{'=' * 50}")
    log_on_main(f"Distributed Training Started")
    log_on_main(f"{'=' * 50}")
    log_on_main(f"Rank: {rank} / {world_size}")
    log_on_main(f"Local Rank: {local_rank}")
    log_on_main(f"Device: {device}")
    log_on_main(f"Mode: {args.mode}")
    log_on_main(f"Strategy: {cfg.strategy}")
    log_on_main(f"{'=' * 50}\n")

    # -------------------------
    # Data (DistributedSampler)
    # -------------------------
    train_dataset = get_cifar100_dataset(train=True)
    test_dataset = get_cifar100_dataset(train=False)

    train_loader = get_distributed_dataloader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        rank=rank,
        world_size=world_size,
        shuffle=True,
    )

    test_loader = get_distributed_dataloader(
        test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        rank=rank,
        world_size=world_size,
        shuffle=False,
    )

    # Get sampler for epoch synchronization
    train_sampler = train_loader.sampler
    test_sampler = test_loader.sampler

    log_on_main(f"Train batches per epoch: {len(train_loader)}")
    log_on_main(f"Test batches: {len(test_loader)}")

    # -------------------------
    # Model
    # -------------------------
    model = get_model(cfg.model, num_classes=100)
    model.to(device)

    # Convert BatchNorm to SyncBatchNorm for multi-GPU training
    # SyncBatchNorm synchronizes batch statistics across all GPUs
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Wrap with DDP for distributed training
    # DDP handles gradient synchronization across GPUs automatically
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # -------------------------
    # Checkpoint Manager
    # -------------------------
    checkpoint_manager = CheckpointManager(is_ddp_wrapped=True)

    bit_range = None
    if args.fault_bit_range:
        low_str, high_str = args.fault_bit_range.split(",", maxsplit=1)
        bit_range = (int(low_str.strip()), int(high_str.strip()))

    target_layers = []
    if args.fault_target_layers.strip():
        target_layers = [
            int(idx.strip())
            for idx in args.fault_target_layers.split(",")
            if idx.strip()
        ]

    fault_cfg = CheckpointFaultConfig(
        enabled=not args.no_fault_injection,
        fault_type=FaultType(args.fault_type),
        fault_location=args.fault_location,
        bit_range=bit_range,
        fault_probability=args.fault_probability,
        num_bit_flips=args.fault_bit_flips,
        num_processes=args.fault_num_processes,
        total_processes=world_size,
        auto_precision=True,
        target_layers=target_layers,
        specific_bit_position=args.fault_specific_bit,
        log_injection=True,
        injection_log_path=args.fault_log_path,
        load_log=args.fault_load_log,
        seed=args.fault_seed,
    )
    checkpoint_injector = CheckpointBitFlipInjector(
        config=fault_cfg,
        rank=rank,
        world_size=world_size,
        dt_mechanism="DDP",
    )

    # Try to resume from checkpoint
    start_epoch, best_val_acc = checkpoint_manager.load_latest(
        model,
        optimizer,
        device,
        checkpoint_injector=checkpoint_injector,
    )
    log_on_main(f"Resuming from epoch {start_epoch}")
    if fault_cfg.enabled:
        log_on_main(
            "Checkpoint fault injection enabled by default | "
            f"type={fault_cfg.fault_type.value}, location={fault_cfg.fault_location}, "
            f"bit_flips={fault_cfg.num_bit_flips}, target_processes={fault_cfg.num_processes}/{world_size}"
        )

    # -------------------------
    # Metrics Collector
    # -------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"distributed_training_{timestamp}"
    
    metrics = MetricsCollector(
        experiment_name=experiment_name,
        training_mode="distributed",
        dataset="cifar100",
        model=cfg.model,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
    )
    
    metrics.set_distributed_config(world_size=world_size)
    log_on_main(f"Metrics collection enabled | experiment: {experiment_name}")

    # -------------------------
    # Training Loop
    # -------------------------
    log_on_main("\nStarting training...")

    for epoch in range(start_epoch, cfg.epochs):
        metrics.start_epoch(epoch)
        epoch_start = time.time()

        # Train one epoch
        avg_loss = train_epoch(
            model, train_loader, optimizer, device, epoch, train_sampler
        )

        # Synchronize after epoch (all workers must complete training)
        if is_distributed_initialized():
            barrier()
            metrics.record_synchronization(sync_time_sec=0.1)  # Approximate

        epoch_time = time.time() - epoch_start

        # Evaluate - get local metrics
        correct, total, local_acc = evaluate(model, test_loader, device)

        # Aggregate metrics across all GPUs
        if is_distributed_initialized():
            avg_loss, avg_acc = reduce_metrics(avg_loss, correct, total, world_size)
        else:
            avg_acc = local_acc

        # Record epoch metrics
        metrics.end_epoch(
            epoch=epoch,
            train_loss=avg_loss,
            train_acc=avg_acc,
            val_loss=avg_loss,
            val_acc=avg_acc
        )

        # Log results (rank 0 only)
        log_on_main(
            f"Epoch {epoch + 1}/{cfg.epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Accuracy: {avg_acc:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

        # Save checkpoint (rank 0 only)
        ckpt_start = time.time()
        checkpoint_manager.save(model, optimizer, epoch + 1, avg_loss)
        ckpt_time = time.time() - ckpt_start
        
        metrics.record_checkpoint(checkpoint_size_mb=0.0, save_time_sec=ckpt_time)

    # -------------------------
    # Cleanup
    # -------------------------
    log_on_main("\nTraining completed!")

    # Export metrics (rank 0 only)
    if is_main_process():
        log_on_main("\nExporting metrics...")
        exporter = MetricsExporter(base_export_dir="./experiment_results")
        
        try:
            json_path = exporter.export_summary_json(metrics)
            log_on_main(f"  ✓ JSON summary: {json_path}")
        except Exception as e:
            log_on_main(f"  ✗ JSON export failed: {e}")
        
        try:
            jsonl_path = exporter.export_summary_jsonl(metrics)
            log_on_main(f"  ✓ JSONL aggregated: {jsonl_path}")
        except Exception as e:
            log_on_main(f"  ✗ JSONL export failed: {e}")
        
        try:
            csv_path = exporter.export_epoch_metrics_csv(metrics)
            log_on_main(f"  ✓ Epoch CSV: {csv_path}")
        except Exception as e:
            log_on_main(f"  ✗ CSV export failed: {e}")
        
        log_on_main(f"\nMetrics saved to: ./experiment_results/{metrics.experiment_name}/")

    if is_distributed_initialized():
        cleanup_distributed()

    return 0


# -------------------------------------------------
# Entry Point
# -------------------------------------------------
if __name__ == "__main__":
    # Required for torchrun to work properly
    # This ensures the main module is not re-executed
    sys.exit(main())
