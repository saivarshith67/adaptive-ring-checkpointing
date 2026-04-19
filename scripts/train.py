import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
import time
import random
from datetime import datetime
import sys
import numpy as np

from src.coci.models.model import get_model
from src.coci.data_ingestor.cifar import get_cifar100_dataset
from src.coci.config import load_config
from src.coci.checkpointing.checkpoint_manager import CheckpointManager
from src.coci.checkpointing.strategy import CheckpointStrategyFactory
from src.coci.metrics import MetricsCollector, MetricsExporter
from src.coci.fault.checkpoint_fault_injector import (
    CheckpointBitFlipInjector,
    CheckpointFaultConfig,
    FaultType,
    enforce_determinism,
)


# -------------------------------------------------
# Training Loop
# -------------------------------------------------
def train(
    model,
    train_loader,
    test_loader,
    optimizer,
    device,
    cfg,
    checkpoint_manager,
    strategy,
    metrics=None,
    fault_injector=None,
    inject_fault=False,
    start_epoch=0,
):

    for epoch in range(start_epoch, cfg.epochs):
        if metrics:
            metrics.start_epoch(epoch)
            
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # Track training accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # -------------------------
            # Time-based checkpointing
            # -------------------------
            if cfg.strategy != "epoch":
                if strategy.should_checkpoint():
                    ckpt_start = time.time()
                    checkpoint_manager.save(model, optimizer, epoch, total_loss)
                    ckpt_time = time.time() - ckpt_start
                    
                    if metrics:
                        metrics.record_batch_checkpoint(batch_id=batch_idx, checkpoint_size_mb=0.0, save_time_sec=ckpt_time)
                    
                    strategy.update_checkpoint_time()

            # -------------------------
            # Fault Injection
            # -------------------------
            if inject_fault and fault_injector is not None:
                fault_injector.maybe_fail()

        epoch_time = time.time() - epoch_start
        train_loss = total_loss / max(1, len(train_loader))
        train_acc = 100.0 * correct / total if total > 0 else 0.0

        print(
            f"Epoch {epoch + 1}/{cfg.epochs} | "
            f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

        accuracy = evaluate(model, test_loader, device)
        print(f"Test Accuracy: {accuracy:.2f}%")

        # Record epoch metrics
        if metrics:
            metrics.end_epoch(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=0.0,  # Not computed separately in this script
                val_acc=accuracy
            )

        # -------------------------
        # Epoch-based checkpointing
        # -------------------------
        if cfg.strategy == "epoch":
            ckpt_start = time.time()
            checkpoint_manager.save(model, optimizer, epoch, train_loss)
            ckpt_time = time.time() - ckpt_start
            
            if metrics:
                metrics.record_checkpoint(checkpoint_size_mb=0.0, save_time_sec=ckpt_time)

    return True


# -------------------------------------------------
# Evaluation
# -------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100.0 * correct / total


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dev", "server"], default="dev")
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
    )
    parser.add_argument("--fault-location", type=str, default="model")
    parser.add_argument("--fault-bit-range", type=str, default=None)
    parser.add_argument("--fault-probability", type=float, default=1.0)
    parser.add_argument("--fault-bit-flips", type=int, default=1)
    parser.add_argument("--fault-num-processes", type=int, default=1)
    parser.add_argument("--fault-target-layers", type=str, default="")
    parser.add_argument("--fault-specific-bit", type=int, default=None)
    parser.add_argument("--fault-log-path", type=str, default=None)
    parser.add_argument("--fault-load-log", type=str, default=None)
    parser.add_argument("--fault-seed", type=int, default=42)

    args = parser.parse_args()

    config_path = "configs/dev.yaml" if args.mode == "dev" else "configs/server.yaml"

    cfg = load_config(config_path)

    print(f"\nRunning in {args.mode.upper()} mode")
    print(f"Using strategy: {cfg.strategy}")

    enforce_determinism(args.fault_seed)
    random.seed(args.fault_seed)
    np.random.seed(args.fault_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # -------------------------
    # Data
    # -------------------------
    train_loader = DataLoader(
        get_cifar100_dataset(train=True),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    test_loader = DataLoader(
        get_cifar100_dataset(train=False),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    # -------------------------
    # Model
    # -------------------------
    model = get_model(cfg.model, num_classes=100)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # -------------------------
    # Checkpoint Manager
    # -------------------------
    checkpoint_manager = CheckpointManager()

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
        total_processes=1,
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
        rank=0,
        world_size=1,
        dt_mechanism="DDP",
    )
    start_epoch, best_val_acc = checkpoint_manager.load_latest(
        model,
        optimizer,
        device,
        checkpoint_injector=checkpoint_injector,
    )

    if start_epoch >= cfg.epochs:
        print("Training already completed.")
        sys.exit(0)

    # -------------------------
    # Strategy Setup
    # -------------------------
    checkpoint_cost = cfg.checkpoint_cost_estimate

    mtbf_estimate = (
        1.0 / cfg.failure_rate_per_second if cfg.failure_rate_per_second > 0 else 1e9
    )

    strategy = CheckpointStrategyFactory.create(
        cfg, checkpoint_cost=checkpoint_cost, mtbf=mtbf_estimate
    )

    # 🔥 Reset timer after resume
    strategy.update_checkpoint_time()

    # -------------------------
    # Metrics Collector
    # -------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"cifar100_{cfg.strategy}_{timestamp}"
    
    metrics = MetricsCollector(
        experiment_name=experiment_name,
        training_mode=cfg.strategy,
        dataset="cifar100",
        model=cfg.model,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
    )
    
    metrics.set_distributed_config(world_size=1)
    print(f"Metrics collection enabled | experiment: {experiment_name}")

    # -------------------------
    # Global Runtime Start
    # -------------------------
    overall_start_time = time.time()

    # -------------------------
    # Train
    # -------------------------
    try:
        train(
            model,
            train_loader,
            test_loader,
            optimizer,
            device,
            cfg,
            checkpoint_manager,
            strategy,
            metrics=metrics,
            start_epoch=start_epoch,
        )

        total_runtime = time.time() - overall_start_time

        failures = 0
        measured_mtbf = None

        summary = {
            "timestamp": datetime.now().isoformat(),
            "strategy": cfg.strategy,
            "total_runtime_sec": total_runtime,
            "num_checkpoints": checkpoint_manager.num_checkpoints,
            "total_checkpoint_time_sec": checkpoint_manager.total_checkpoint_time,
            "failures": failures,
            "measured_mtbf": measured_mtbf,
        }

        print("\n========== EXPERIMENT SUMMARY ==========")
        for k, v in summary.items():
            print(f"{k}: {v}")
        print("========================================\n")

        with open(f"final_experiment_log_{cfg.strategy}.jsonl", "a") as f:
            f.write(json.dumps(summary) + "\n")

        # Export metrics
        print("\nExporting metrics...")
        exporter = MetricsExporter(base_export_dir="./experiment_results")
        
        try:
            json_path = exporter.export_summary_json(metrics)
            print(f"  ✓ JSON summary: {json_path}")
        except Exception as e:
            print(f"  ✗ JSON export failed: {e}")
        
        try:
            jsonl_path = exporter.export_summary_jsonl(metrics)
            print(f"  ✓ JSONL aggregated: {jsonl_path}")
        except Exception as e:
            print(f"  ✗ JSONL export failed: {e}")
        
        try:
            csv_path = exporter.export_epoch_metrics_csv(metrics)
            print(f"  ✓ Epoch CSV: {csv_path}")
        except Exception as e:
            print(f"  ✗ CSV export failed: {e}")
        
        print(f"\nMetrics saved to: ./experiment_results/{metrics.experiment_name}/")

        print("Training finished successfully.")
        sys.exit(0)

    except RuntimeError as e:
        print(f"\n💥 Training interrupted: {e}")

        sys.exit(1)


if __name__ == "__main__":
    main()
