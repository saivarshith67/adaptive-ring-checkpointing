#!/usr/bin/env python3
"""
FaceForensics++ Deepfake Detection Training Script

A unified training script for deepfake detection on the FaceForensics++ dataset.
Supports both single-GPU and multi-GPU (via torchrun) training.

Key Features (integrated from Kaggle notebook patterns):
    - Video-level processing with multi-frame sampling
    - CNN backbone with temporal aggregation (LSTM/GRU or mean pooling)
    - Efficient video loading with pre-extracted frame caching
    - Multi-GPU support via DistributedDataParallel (DDP)

Usage:
    # Single GPU
    python scripts/train_faceforensics.py

    # Multi-GPU (recommended)
    torchrun --nproc_per_node=2 scripts/train_faceforensics.py

    # Multi-GPU with 4 GPUs
    torchrun --nproc_per_node=4 scripts/train_faceforensics.py

    # With custom settings
    python scripts/train_faceforensics.py --epochs 50 --batch_size 32 --lr 1e-4

    # Download dataset first
    python scripts/train_faceforensics.py --download-dataset

    # Resume from checkpoint
    python scripts/train_faceforensics.py --resume

    # Use temporal model (LSTM aggregation)
    python scripts/train_faceforensics.py --temporal-model lstm --num-frames 16

    # Use mean pooling (faster, single-frame equivalent)
    python scripts/train_faceforensics.py --temporal-model mean
"""

import argparse
import copy
from datetime import datetime
import os
import sys
import time
import json
import random
from pathlib import Path
from dataclasses import replace
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
from src.coci.models.model import get_model, get_multiframe_model
from src.coci.data_ingestor.faceforensics import (
    FaceForensicsDataset,
    VideoFaceForensicsDataset,
    VideoDatasetFast,
    download_faceforensics_dataset,
    get_faceforensics_transforms,
    precompute_face_crops,
    preextract_frames,
)
from src.coci.checkpointing.checkpoint_manager import CheckpointManager
from src.coci.checkpointing.hash_ring_checkpoint_manager import HashRingCheckpointManager
from src.coci.checkpointing.convergence_scheduler import ConvergenceAwareScheduler
from src.coci.hashing import create_hash_ring, HashRing
from src.coci.metrics import MetricsCollector, MetricsExporter
from src.coci.fault.checkpoint_fault_injector import (
    CheckpointBitFlipInjector,
    CheckpointFaultConfig,
    FaultType,
    enforce_determinism,
)
from src.coci.config import (
    FACEFORENSICS_DATASET_PATH,
    MODEL_NAME,
    NUM_CLASSES,
    FACEFORENSICS_COMPRESSION,
)


# -------------------------------------------------
# Configuration Defaults
# -------------------------------------------------
DEFAULT_CONFIG = {
    "dataset_path": FACEFORENSICS_DATASET_PATH,
    "compression": FACEFORENSICS_COMPRESSION,
    "model": MODEL_NAME,
    "num_classes": NUM_CLASSES,
    "epochs": 20,
    "batch_size": 8,  # Reduced for video processing (multiple frames)
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "num_workers": 4,
    "checkpoint_dir": "./checkpoints",
    "checkpoint_interval": 5,  # Save every N epochs
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # Video processing settings
    "num_frames": 8,  # Frames to sample per video
    "temporal_model": "mean",  # Aggregation: "mean", "lstm", "gru", "attention"
}


TRAINING_MODE_EPOCH = "epoch"
TRAINING_MODE_CONVERGENCE = "convergence"
TRAINING_MODE_HASH_RING_EPOCH = "hash-ring-epoch"
TRAINING_MODE_CONVERGENCE_HASH_RING = "convergence-hash-ring"
RECOVERY_CONTEXT_FILENAME = "recovery_context.json"


# -------------------------------------------------
# Worker Seeding
# -------------------------------------------------
def seed_worker(worker_id):
    """Seed worker for reproducible data loading."""
    worker_seed = torch.initial_seed() % 2**32
    import numpy as np
    import random

    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def recovery_context_path(checkpoint_dir: str) -> str:
    """Return the path to the persisted recovery-context sidecar."""
    return os.path.join(checkpoint_dir, RECOVERY_CONTEXT_FILENAME)


def load_recovery_context(checkpoint_dir: str) -> Optional[dict]:
    """Load persisted recovery context if present."""
    path = recovery_context_path(checkpoint_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def save_recovery_context(checkpoint_dir: str, payload: dict) -> None:
    """Persist recovery context across crash/restart cycles."""
    path = recovery_context_path(checkpoint_dir)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def clear_recovery_context(checkpoint_dir: str) -> None:
    """Remove persisted recovery context after a successful resume."""
    path = recovery_context_path(checkpoint_dir)
    if os.path.exists(path):
        os.remove(path)


def parse_step_from_checkpoint_id(checkpoint_id: Optional[str]) -> int:
    """Best-effort extraction of a global step from a checkpoint identifier."""
    if not checkpoint_id:
        return 0
    if checkpoint_id.startswith("step_"):
        try:
            return int(checkpoint_id.split("_", maxsplit=1)[1])
        except ValueError:
            return 0
    return 0


# -------------------------------------------------
# Dataset Setup
# -------------------------------------------------
def get_faceforensics_dataloader(
    dataset_path: str,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
    compression: str = "c23",
    limit: int = None,
    train: bool = True,
    use_precropped: bool = True,
    video_mode: bool = False,
    num_frames: int = 8,
):
    """
    Create DataLoader with DistributedSampler for FaceForensics++ dataset.

    Args:
        dataset_path: Path to FaceForensics++ dataset
        batch_size: Batch size per process
        num_workers: Number of data loading workers
        rank: Current process rank
        world_size: Total number of processes
        compression: Compression level (c23 or c40)
        limit: Limit number of samples (for development)
        train: Whether this is training (enables shuffling)
        use_precropped: Use pre-computed face crops (recommended)
        video_mode: If True, use video-level dataset (multiple frames per video)
        num_frames: Number of frames to sample per video (for video_mode)

    Returns:
        DataLoader with DistributedSampler
    """
    cache_dir = os.path.join(dataset_path, "crops")

    if video_mode:
        # Video-level dataset: groups frames by video
        dataset = VideoFaceForensicsDataset(
            root=dataset_path,
            compression=compression,
            limit=limit,
            use_precropped=use_precropped,
            cache_dir=cache_dir,
            num_frames=num_frames,
        )
    else:
        # Image-level dataset: individual frames
        transform = get_faceforensics_transforms(include_resize=not use_precropped)
        dataset = FaceForensicsDataset(
            root=dataset_path,
            transform=transform,
            compression=compression,
            limit=limit,
            use_precropped=use_precropped,
            cache_dir=cache_dir,
        )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=train,
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

    return dataloader, sampler


# -------------------------------------------------
# Training Functions
# -------------------------------------------------
def train_epoch(
    model,
    train_loader,
    optimizer,
    device,
    epoch,
    sampler=None,
    video_mode=False,
    checkpoint_callback: Optional[Callable[[int, float], None]] = None,
):
    """
    Train for one epoch.

    Args:
        model: The neural network model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        sampler: DistributedSampler for epoch synchronization
        video_mode: If True, expects (frames, labels) where frames is (B, T, C, H, W)
    Returns:
        tuple: (avg_loss, correct, total) - local metrics
    """
    model.train()

    # Synchronize epoch for proper data shuffling across epochs
    if sampler is not None:
        sampler.set_epoch(epoch)

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(train_loader):
        if video_mode:
            # Video mode: batch is (frames, labels) where frames is (B, T, C, H, W)
            frames, labels = batch
            frames = frames.to(device)
            labels = labels.to(device)

            # frames shape: (batch_size, num_frames, channels, height, width)
            batch_size = frames.shape[0]
            num_frames = frames.shape[1]

            # Reshape for model: (batch_size * num_frames, channels, height, width)
            frames = frames.view(-1, *frames.shape[2:])

            optimizer.zero_grad()
            outputs = model(frames)  # Shape: (batch_size * num_frames, num_classes)

            # Reshape outputs: (batch_size, num_frames, num_classes)
            outputs = outputs.view(batch_size, num_frames, -1)

            # Aggregate frame predictions (mean across frames)
            outputs = outputs.mean(dim=1)  # Shape: (batch_size, num_classes)
        else:
            # Image mode: batch is (images, labels)
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Track accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if checkpoint_callback is not None:
            checkpoint_callback(batch_idx, loss.item())

        # Log progress every 50 batches
        if batch_idx % 50 == 0 and is_main_process():
            log_on_main(
                f"  Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    return avg_loss, correct, total


def evaluate(model, test_loader, device, video_mode=False):
    """
    Evaluate the model on test/validation data.

    Args:
        model: The neural network model
        test_loader: Test/validation data loader
        device: Device to evaluate on
        video_mode: If True, expects (frames, labels) where frames is (B, T, C, H, W)

    Returns:
        tuple: (avg_loss, correct, total, accuracy) - local metrics
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            if video_mode:
                frames, labels = batch
                frames = frames.to(device)
                labels = labels.to(device)

                batch_size = frames.shape[0]
                num_frames = frames.shape[1]

                frames = frames.view(-1, *frames.shape[2:])
                outputs = model(frames)
                outputs = outputs.view(batch_size, num_frames, -1).mean(dim=1)
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

            loss = F.cross_entropy(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, correct, total, accuracy


# -------------------------------------------------
# Dataset Download
# -------------------------------------------------
def download_and_prepare_dataset():
    """Download FaceForensics++ dataset using kagglehub."""
    log_on_main("\n" + "=" * 60)
    log_on_main("Downloading FaceForensics++ Dataset")
    log_on_main("=" * 60)
    log_on_main("Kaggle slug: hungle3401/faceforensics")
    log_on_main("This may take a while on first run...\n")

    try:
        path = download_faceforensics_dataset()
        log_on_main(f"\n✓ Dataset downloaded to: {path}")
        log_on_main(f"\nYou can now run training with:")
        log_on_main(f"  python scripts/train_faceforensics.py --dataset-path {path}")
        return path
    except Exception as e:
        log_on_main(f"\n✗ Download failed: {e}")
        log_on_main("\nTo download manually:")
        log_on_main("1. Go to https://www.kaggle.com/datasets/hungle3401/faceforensics")
        log_on_main("2. Download and extract to ./data/faceforensics")
        return None


# -------------------------------------------------
# Main Training
# -------------------------------------------------
def main(cli_args=None):
    parser = argparse.ArgumentParser(
        description="FaceForensics++ Deepfake Detection Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings (single GPU)
  python scripts/train_faceforensics.py

  # Train with 2 GPUs
  torchrun --nproc_per_node=2 scripts/train_faceforensics.py

  # Train with 4 GPUs on multiple nodes
  torchrun --nproc_per_node=4 --nnodes=2 scripts/train_faceforensics.py

  # Download dataset first
  python scripts/train_faceforensics.py --download-dataset

  # Resume from checkpoint
  python scripts/train_faceforensics.py --resume

  # Development mode with limited data
  python scripts/train_faceforensics.py --limit 1000 --epochs 10
        """,
    )

    # Dataset arguments
    parser.add_argument(
        "--download-dataset",
        action="store_true",
        help="Download FaceForensics++ dataset via kagglehub and exit",
    )
    parser.add_argument(
        "--precrop",
        action="store_true",
        default=True,
        help="Pre-compute face crops before training (recommended, default: True)",
    )
    parser.add_argument(
        "--no-precrop",
        action="store_true",
        help="Skip pre-computing face crops (slower but uses less disk space)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DEFAULT_CONFIG["dataset_path"],
        help=f"Path to FaceForensics++ dataset (default: {DEFAULT_CONFIG['dataset_path']})",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default=DEFAULT_CONFIG["compression"],
        choices=["c23", "c40"],
        help=f"Compression level (c23=visually lossless, c40=highly compressed, default: {DEFAULT_CONFIG['compression']})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples for development (default: all)",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="Train/val split ratio (default: 0.8 = 80%% train)",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_CONFIG["model"],
        choices=["efficientnet_b0", "resnet18", "resnet50", "mobilenet"],
        help=f"Model architecture (default: {DEFAULT_CONFIG['model']})",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Don't use pretrained ImageNet weights",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_CONFIG["epochs"],
        help=f"Number of training epochs (default: {DEFAULT_CONFIG['epochs']})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_CONFIG["batch_size"],
        help=f"Batch size per GPU (default: {DEFAULT_CONFIG['batch_size']})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_CONFIG["lr"],
        help=f"Learning rate (default: {DEFAULT_CONFIG['lr']})",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_CONFIG["weight_decay"],
        help=f"Weight decay (default: {DEFAULT_CONFIG['weight_decay']})",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_CONFIG["num_workers"],
        help=f"DataLoader workers (default: {DEFAULT_CONFIG['num_workers']})",
    )

    # Video processing arguments (integrated from Kaggle notebook patterns)
    parser.add_argument(
        "--video-mode",
        action="store_true",
        help="Process videos at video-level (sample multiple frames per video)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_CONFIG["num_frames"],
        help=f"Number of frames to sample per video in video mode (default: {DEFAULT_CONFIG['num_frames']})",
    )
    parser.add_argument(
        "--temporal-model",
        type=str,
        default=DEFAULT_CONFIG["temporal_model"],
        choices=["mean", "lstm", "gru", "attention"],
        help=f"Temporal aggregation method (default: {DEFAULT_CONFIG['temporal_model']})",
    )
    parser.add_argument(
        "--fast-video-mode",
        action="store_true",
        help="Use pre-extracted .npy frames for fastest loading (run --extract-frames first)",
    )
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="Pre-extract frames from videos to .npy files for fast loading",
    )

    # Legacy runtime fault arguments (kept for CLI compatibility)
    parser.add_argument(
        "--inject-fault",
        action="store_true",
        help="Deprecated: use checkpoint fault injection options (enabled by default)",
    )
    parser.add_argument(
        "--inject-rank",
        type=int,
        default=0,
        help="Deprecated: retained for backward compatibility",
    )
    parser.add_argument(
        "--inject-rate",
        type=float,
        default=0.02,
        help="Deprecated: retained for backward compatibility",
    )

    # Checkpoint bit-flip fault injection arguments
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

    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=DEFAULT_CONFIG["checkpoint_dir"],
        help=f"Checkpoint directory (default: {DEFAULT_CONFIG['checkpoint_dir']})",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CONFIG["checkpoint_interval"],
        help=f"Save checkpoint every N epochs (default: {DEFAULT_CONFIG['checkpoint_interval']})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint",
    )
    parser.add_argument(
        "--training-mode",
        type=str,
        default=None,
        choices=[
            TRAINING_MODE_EPOCH,
            TRAINING_MODE_CONVERGENCE,
            TRAINING_MODE_HASH_RING_EPOCH,
            TRAINING_MODE_CONVERGENCE_HASH_RING,
        ],
        help=(
            "Checkpoint behavior mode. "
            "epoch=epoch checkpoints only, "
            "convergence=COCI timing + normal storage, "
            "hash-ring-epoch=epoch timing + hash ring storage, "
            "convergence-hash-ring=COCI timing + hash ring storage"
        ),
    )
    parser.add_argument(
        "--checkpoint-mode",
        type=str,
        default="normal",
        choices=["normal", "hash-ring"],
        help="Legacy backend selector (default: normal)",
    )
    parser.add_argument(
        "--hash-ring-virtual-nodes",
        type=int,
        default=100,
        help="Virtual nodes per physical node for hash ring mode (default: 100)",
    )
    parser.add_argument(
        "--hash-ring-cache-dir",
        type=str,
        default="./checkpoints/hash_cache",
        help="Local cache root for hash ring checkpoint shards",
    )
    parser.add_argument(
        "--failure-rate-lambda",
        type=float,
        default=0.00208,
        help="Failure rate lambda (1/MTBF) for convergence-aware checkpointing",
    )
    parser.add_argument(
        "--checkpoint-cost-sec",
        type=float,
        default=0.60,
        help="Estimated checkpoint save cost in seconds for convergence-aware checkpointing",
    )
    parser.add_argument(
        "--auto-calibrate-checkpoint-cost",
        dest="auto_calibrate_checkpoint_cost",
        action="store_true",
        default=True,
        help="Enable runtime checkpoint-cost auto-calibration from measured save times (default: enabled)",
    )
    parser.add_argument(
        "--no-auto-calibrate-checkpoint-cost",
        dest="auto_calibrate_checkpoint_cost",
        action="store_false",
        help="Disable runtime checkpoint-cost auto-calibration",
    )
    parser.add_argument(
        "--checkpoint-cost-warmup-saves",
        type=int,
        default=3,
        help="Number of early convergence checkpoints used to calibrate checkpoint cost (default: 3)",
    )
    parser.add_argument(
        "--checkpoint-cost-ema-alpha",
        type=float,
        default=0.5,
        help="EMA smoothing factor for checkpoint-cost calibration in (0, 1] (default: 0.5)",
    )
    parser.add_argument(
        "--fit-interval-steps",
        type=int,
        default=100,
        help="Batch interval for online loss fitting in convergence mode",
    )
    parser.add_argument(
        "--min-convergence-interval-sec",
        type=float,
        default=5.0,
        help="Minimum interval between convergence checkpoints in seconds",
    )
    parser.add_argument(
        "--max-convergence-interval-sec",
        type=float,
        default=1800.0,
        help="Maximum interval between convergence checkpoints in seconds",
    )
    parser.add_argument(
        "--runtime-fault-injection",
        action="store_true",
        help="Inject a live fault during training after checkpoints are created",
    )
    parser.add_argument(
        "--runtime-fault-after-checkpoints",
        type=int,
        default=1,
        help="Inject the live fault after this many successful checkpoint saves (default: 1)",
    )
    parser.add_argument(
        "--runtime-fault-crash-after-injection",
        dest="runtime_fault_crash_after_injection",
        action="store_true",
        default=True,
        help="Crash the job after applying the live fault so resume/recovery is exercised (default: True)",
    )
    parser.add_argument(
        "--no-runtime-fault-crash-after-injection",
        dest="runtime_fault_crash_after_injection",
        action="store_false",
        help="Keep training running after the live fault is injected",
    )

    args = parser.parse_args(cli_args)

    if args.checkpoint_cost_warmup_saves < 1:
        parser.error("--checkpoint-cost-warmup-saves must be >= 1")
    if not (0.0 < args.checkpoint_cost_ema_alpha <= 1.0):
        parser.error("--checkpoint-cost-ema-alpha must be in (0, 1]")
    if args.runtime_fault_after_checkpoints < 1:
        parser.error("--runtime-fault-after-checkpoints must be >= 1")

    if args.training_mode is None:
        if args.checkpoint_mode == "hash-ring":
            active_training_mode = TRAINING_MODE_HASH_RING_EPOCH
        else:
            active_training_mode = TRAINING_MODE_EPOCH
    else:
        active_training_mode = args.training_mode

    use_hash_ring = active_training_mode in {
        TRAINING_MODE_HASH_RING_EPOCH,
        TRAINING_MODE_CONVERGENCE_HASH_RING,
    }
    use_convergence = active_training_mode in {
        TRAINING_MODE_CONVERGENCE,
        TRAINING_MODE_CONVERGENCE_HASH_RING,
    }

    # Handle dataset download
    if args.download_dataset:
        download_and_prepare_dataset()
        return 0

    # Handle frame extraction
    if args.extract_frames:
        log_on_main("\n" + "=" * 60)
        log_on_main("Pre-extracting frames from videos")
        log_on_main("=" * 60)
        frames_output_dir = os.path.join(args.dataset_path, "frames")
        preextract_frames(
            root_dir=args.dataset_path,
            output_dir=frames_output_dir,
            num_frames=args.num_frames,
        )
        log_on_main(f"\n✓ Frames extracted to: {frames_output_dir}")
        log_on_main("\nYou can now run training with:")
        log_on_main(
            f"  python scripts/train_faceforensics.py --fast-video-mode --dataset-path {args.dataset_path}"
        )
        return 0

    # -------------------------
    # Distributed Setup
    # -------------------------
    rank, world_size, local_rank = setup_distributed(backend="nccl")
    enforce_determinism(args.fault_seed)
    random.seed(args.fault_seed + rank)
    torch.manual_seed(args.fault_seed + rank)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_distributed_initialized() and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

    # -------------------------
    # Verify Dataset
    # -------------------------
    if not os.path.exists(args.dataset_path):
        log_on_main(f"\n✗ Dataset not found at: {args.dataset_path}")
        log_on_main("\nDownload the dataset first:")
        log_on_main("  python scripts/train_faceforensics.py --download-dataset")
        log_on_main("\nOr manually download from:")
        log_on_main("  https://www.kaggle.com/datasets/hungle3401/faceforensics")

        if is_distributed_initialized():
            cleanup_distributed()
        return 1

    # -------------------------
    # Logging Header
    # -------------------------
    log_on_main("\n" + "=" * 70)
    log_on_main("FaceForensics++ Deepfake Detection Training")
    log_on_main("=" * 70)
    log_on_main(f"Model:         {args.model}")
    log_on_main(f"Dataset:       {args.dataset_path}")
    log_on_main(f"Compression:   {args.compression}")
    log_on_main(f"Epochs:        {args.epochs}")
    log_on_main(f"Batch Size:    {args.batch_size} (per GPU)")
    log_on_main(f"Learning Rate: {args.lr}")
    log_on_main(f"Checkpoint:    {args.checkpoint_dir}")
    log_on_main(f"Training Mode: {active_training_mode}")
    log_on_main(f"CheckpointMode:{'hash-ring' if use_hash_ring else 'normal'}")
    log_on_main("-" * 70)
    log_on_main(f"Video Mode:    {args.video_mode or args.fast_video_mode}")
    if args.video_mode or args.fast_video_mode:
        log_on_main(f"Frames/Video:  {args.num_frames}")
        if args.fast_video_mode:
            log_on_main(f"Fast Mode:     Pre-extracted .npy frames")
        else:
            log_on_main(f"Temporal Agg:  {args.temporal_model}")
    log_on_main("-" * 70)
    log_on_main(f"Rank:          {rank} / {world_size}")
    log_on_main(f"Device:        {device}")
    log_on_main(f"Pretrained:    {not args.no_pretrained}")
    log_on_main("=" * 70 + "\n")

    # -------------------------
    # Pre-compute Face Crops (recommended for speed)
    # -------------------------
    use_precropped = not args.no_precrop
    cache_dir = os.path.join(args.dataset_path, "crops")

    if use_precropped and not os.path.exists(os.path.join(cache_dir, "manifest.json")):
        log_on_main("\nPre-computing face crops (one-time setup)...")
        log_on_main("This may take a while but makes training much faster.")
        precompute_face_crops(
            dataset_root=args.dataset_path,
            cache_dir=cache_dir,
            compression=args.compression,
            batch_size=64,
            device=str(device),
        )

    # -------------------------
    # Create Datasets
    # -------------------------
    log_on_main("Loading FaceForensics++ dataset...")

    if args.fast_video_mode:
        # Fastest loading mode: use pre-extracted .npy frames
        frames_dir = os.path.join(args.dataset_path, "frames")

        if not os.path.exists(frames_dir):
            log_on_main(f"\n✗ Extracted frames not found at: {frames_dir}")
            log_on_main("\nRun frame extraction first:")
            log_on_main(
                f"  python scripts/train_faceforensics.py --extract-frames --dataset-path {args.dataset_path}"
            )
            if is_distributed_initialized():
                cleanup_distributed()
            return 1

        full_dataset = VideoDatasetFast(
            root_dir=frames_dir,
            num_frames=args.num_frames,
        )

        dataset_size = len(full_dataset)
        train_size = int(args.split * dataset_size)
        val_size = dataset_size - train_size

        torch.manual_seed(42)
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

        log_on_main(f"Fast video dataset loaded: {dataset_size} videos")
        log_on_main(f"  Frames per video: {args.num_frames}")
        log_on_main(f"  Train: {train_size} | Val: {val_size}")

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=42,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            worker_init_fn=seed_worker,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            worker_init_fn=seed_worker,
        )
        val_loader_simple = val_loader

        log_on_main(
            f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}"
        )

    elif args.video_mode:
        # Video-level dataset: groups frames by video
        full_dataset = VideoFaceForensicsDataset(
            root=args.dataset_path,
            compression=args.compression,
            limit=args.limit,
            use_precropped=use_precropped,
            cache_dir=cache_dir,
            num_frames=args.num_frames,
        )

        # Video dataset handles its own train/val split internally
        log_on_main(f"Video dataset loaded: {len(full_dataset)} videos")
        log_on_main(f"  Frames per video: {args.num_frames}")

        # Create distributed dataloaders for video mode
        train_loader, train_sampler = get_faceforensics_dataloader(
            args.dataset_path,
            args.batch_size,
            args.num_workers,
            rank,
            world_size,
            compression=args.compression,
            limit=args.limit,
            train=True,
            video_mode=True,
            num_frames=args.num_frames,
        )

        val_loader, val_sampler = get_faceforensics_dataloader(
            args.dataset_path,
            args.batch_size,
            args.num_workers,
            rank,
            world_size,
            compression=args.compression,
            limit=args.limit,
            train=False,
            video_mode=True,
            num_frames=args.num_frames,
        )

        # For validation, use the same sampler for simple loader
        val_loader_simple = val_loader

        log_on_main(f"Train videos: {len(train_loader)} batches")
        log_on_main(f"Val videos:   {len(val_loader)} batches")

    else:
        # Image-level dataset: individual frames
        transform = get_faceforensics_transforms(include_resize=not use_precropped)

        full_dataset = FaceForensicsDataset(
            root=args.dataset_path,
            transform=transform,
            compression=args.compression,
            limit=args.limit,
            use_precropped=use_precropped,
            cache_dir=cache_dir,
        )

        dataset_size = len(full_dataset)
        train_size = int(args.split * dataset_size)
        val_size = dataset_size - train_size

        # Split dataset
        torch.manual_seed(42)
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

        log_on_main(f"Dataset loaded: {dataset_size} samples")
        log_on_main(f"  Train: {train_size} | Val: {val_size}")

        # Create distributed dataloaders
        train_loader, train_sampler = get_faceforensics_dataloader(
            args.dataset_path,
            args.batch_size,
            args.num_workers,
            rank,
            world_size,
            compression=args.compression,
            limit=args.limit,
            train=True,
        )

        val_loader, val_sampler = get_faceforensics_dataloader(
            args.dataset_path,
            args.batch_size,
            args.num_workers,
            rank,
            world_size,
            compression=args.compression,
            limit=args.limit,
            train=False,
        )

        # Create val_sampler for non-distributed use
        val_sampler_simple = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=42,
        )

        val_loader_simple = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler_simple,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

        log_on_main(
            f"Train batches: {len(train_loader)} | Val batches: {len(val_loader_simple)}"
        )

    # -------------------------
    # Model
    # -------------------------
    log_on_main("\nInitializing model...")

    num_classes = 2  # Binary classification: real=0, fake=1

    if args.video_mode:
        # Multi-frame video model with temporal aggregation
        model = get_multiframe_model(
            backbone=args.model,
            temporal_mode=args.temporal_model,
            num_classes=num_classes,
            pretrained=not args.no_pretrained,
            hidden_size=256,
            num_layers=1,
        )
        log_on_main(
            f"Video model initialized: {args.model} + {args.temporal_model} aggregation"
        )
    else:
        # Standard single-frame model
        model = get_model(
            args.model,
            num_classes=num_classes,
            pretrained=not args.no_pretrained,
        )
        log_on_main(f"Model initialized: {args.model} with {num_classes} classes")

    model.to(device)

    # SyncBatchNorm for multi-GPU
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # DDP wrapper
    model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_on_main(f"Trainable parameters: {num_params:,}")

    # -------------------------
    # Optimizer & Scheduler
    # -------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # -------------------------
    # Checkpoint Manager
    # -------------------------
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if use_hash_ring:
        node_ids = [f"rank-{i}" for i in range(world_size)]
        checkpoint_manager = HashRingCheckpointManager(
            checkpoint_dir=args.checkpoint_dir,
            is_ddp_wrapped=True,
            node_id=f"rank-{rank}",
            all_node_ids=node_ids,
            cache_root=args.hash_ring_cache_dir,
            virtual_nodes=args.hash_ring_virtual_nodes,
        )
        log_on_main(
            f"Hash ring checkpointing enabled | cache_root={args.hash_ring_cache_dir} | vnodes={args.hash_ring_virtual_nodes}"
        )
    else:
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=args.checkpoint_dir,
            is_ddp_wrapped=True,
        )

    # Start heartbeat thread for hash-ring systems
    if use_hash_ring:
        checkpoint_manager.start_heartbeat_thread(rank=rank, world_size=world_size)
        log_on_main(
            "Hash-ring heartbeat thread started "
            f"(rank={rank}, world_size={world_size})"
        )

    convergence_scheduler = None
    calibrated_checkpoint_cost_sec = float(args.checkpoint_cost_sec)
    checkpoint_cost_calibration_count = 0
    if use_convergence:
        convergence_scheduler = ConvergenceAwareScheduler(
            failure_rate_lambda=args.failure_rate_lambda,
            checkpoint_cost_seconds=args.checkpoint_cost_sec,
            fit_interval_steps=args.fit_interval_steps,
            min_interval_seconds=args.min_convergence_interval_sec,
            max_interval_seconds=args.max_convergence_interval_sec,
        )
        log_on_main(
            "Convergence-aware checkpointing enabled "
            f"(lambda={args.failure_rate_lambda}, cost={args.checkpoint_cost_sec}s)"
        )
        if args.auto_calibrate_checkpoint_cost:
            log_on_main(
                "Checkpoint-cost auto-calibration enabled "
                f"(warmup_saves={args.checkpoint_cost_warmup_saves}, "
                f"ema_alpha={args.checkpoint_cost_ema_alpha})"
            )

    # -------------------------
    # Metrics Collector
    # -------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"faceforensics_{active_training_mode.replace('-', '_')}_{timestamp}"
    
    metrics = MetricsCollector(
        experiment_name=experiment_name,
        training_mode=active_training_mode,
        dataset="faceforensics",
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    
    metrics.set_distributed_config(world_size=world_size)
    log_on_main(f"Metrics collection enabled | experiment: {experiment_name}")

    def sync_hash_ring_metrics() -> None:
        runtime_metrics = getattr(checkpoint_manager, "get_runtime_metrics", lambda: {})()
        if not runtime_metrics:
            return
        metrics.update_hash_ring_metrics(
            total_shards=runtime_metrics.get("total_shards", 0),
            cache_size_mb=runtime_metrics.get("cache_size_mb", 0.0),
            max_cache_mb=runtime_metrics.get("max_cache_size_mb", 0.0),
            shards_owned=runtime_metrics.get("shards_owned", 0),
        )
        metrics.current_hash_ring.local_cache_hits = runtime_metrics.get(
            "local_cache_hits", metrics.current_hash_ring.local_cache_hits
        )
        metrics.current_hash_ring.local_cache_misses = runtime_metrics.get(
            "local_cache_misses", metrics.current_hash_ring.local_cache_misses
        )
        metrics.current_hash_ring.local_cache_loads = runtime_metrics.get(
            "local_cache_loads", metrics.current_hash_ring.local_cache_loads
        )
        metrics.current_hash_ring.central_storage_loads = runtime_metrics.get(
            "central_storage_loads", metrics.current_hash_ring.central_storage_loads
        )
        metrics.current_hash_ring.cache_write_count = runtime_metrics.get(
            "cache_write_count", metrics.current_hash_ring.cache_write_count
        )
        metrics.current_hash_ring.orphaned_shards = runtime_metrics.get(
            "orphaned_shards", metrics.current_hash_ring.orphaned_shards
        )
        metrics.current_hash_ring.reassigned_shards = runtime_metrics.get(
            "reassigned_shards", metrics.current_hash_ring.reassigned_shards
        )
        metrics.current_hash_ring.recached_shards = runtime_metrics.get(
            "recached_shards", metrics.current_hash_ring.recached_shards
        )
        metrics.current_hash_ring.shard_recovery_count = runtime_metrics.get(
            "shard_recovery_count", metrics.current_hash_ring.shard_recovery_count
        )
        metrics.current_hash_ring.recovery_time_sec = runtime_metrics.get(
            "recovery_time_sec", metrics.current_hash_ring.recovery_time_sec
        )
        metrics.current_hash_ring.last_load_source = runtime_metrics.get(
            "load_source", metrics.current_hash_ring.last_load_source
        )

    sync_hash_ring_metrics()

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

    runtime_fault_saves_seen = 0
    runtime_fault_triggered = False
    pending_recovery_context = load_recovery_context(args.checkpoint_dir)

    def maybe_inject_runtime_fault(checkpoint_label: str) -> None:
        nonlocal runtime_fault_saves_seen, runtime_fault_triggered
        should_raise = False

        if args.runtime_fault_injection and not runtime_fault_triggered:
            if is_main_process():
                runtime_fault_saves_seen += 1
                if runtime_fault_saves_seen >= args.runtime_fault_after_checkpoints:
                    runtime_fault_cfg = replace(
                        fault_cfg,
                        enabled=True,
                        num_processes=world_size,
                    )
                    runtime_injector = CheckpointBitFlipInjector(
                        config=runtime_fault_cfg,
                        rank=rank,
                        world_size=world_size,
                        dt_mechanism="runtime",
                    )

                    model_state = copy.deepcopy(
                        model.module.state_dict()
                        if isinstance(model, DDP)
                        else model.state_dict()
                    )
                    optimizer_state = copy.deepcopy(optimizer.state_dict())
                    runtime_checkpoint = {
                        "epoch": epoch + 1,
                        "model_state_dict": model_state,
                        "optimizer_state_dict": optimizer_state,
                        "loss": 0.0,
                        "checkpoint_id": f"runtime_{checkpoint_label}",
                    }
                    runtime_path = os.path.join(
                        args.checkpoint_dir, f"runtime_fault_{checkpoint_label}.pt"
                    )
                    runtime_injector.inject(runtime_checkpoint, runtime_path)

                    load_target = model.module if isinstance(model, DDP) else model
                    load_target.load_state_dict(runtime_checkpoint["model_state_dict"])
                    optimizer.load_state_dict(runtime_checkpoint["optimizer_state_dict"])

                    runtime_fault_triggered = True
                    fault_wall_time = time.time()
                    metrics.record_runtime_fault(
                        checkpoint_id=checkpoint_label,
                        epoch=epoch + 1,
                        global_step=global_step,
                        wall_time_sec=fault_wall_time,
                    )
                    save_recovery_context(
                        args.checkpoint_dir,
                        {
                            "checkpoint_label": checkpoint_label,
                            "fault_epoch": epoch + 1,
                            "fault_global_step": global_step,
                            "fault_wall_time_sec": fault_wall_time,
                            "runtime_fault_after_checkpoints": runtime_fault_saves_seen,
                            "crash_after_injection": args.runtime_fault_crash_after_injection,
                            "training_mode": active_training_mode,
                        },
                    )
                    log_on_main(
                        "  [Runtime Fault] Injected live checkpoint fault "
                        f"after {runtime_fault_saves_seen} saved checkpoint(s) "
                        f"at {checkpoint_label} | crash_after_injection={args.runtime_fault_crash_after_injection}"
                    )
                    should_raise = args.runtime_fault_crash_after_injection

            if is_distributed_initialized():
                crash_flag = torch.tensor(
                    [1 if should_raise else 0],
                    device=device,
                    dtype=torch.int32,
                )
                torch.distributed.broadcast(crash_flag, src=0)
                should_raise = bool(crash_flag.item())

        if should_raise:
            raise RuntimeError(
                "Injected runtime fault for fault-tolerance recovery test"
            )

    # Resume from checkpoint
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        metrics.record_resume_attempt()
        start_epoch, best_val_acc = checkpoint_manager.load_latest(
            model,
            optimizer,
            device,
            checkpoint_injector=checkpoint_injector,
        )
        load_metadata = getattr(checkpoint_manager, "last_load_metadata", None) or {}
        load_source = load_metadata.get("load_source", "central-storage")
        metrics.record_hash_ring_load(load_source)
        if pending_recovery_context:
            time_to_resume_sec = max(
                0.0,
                time.time() - pending_recovery_context.get("fault_wall_time_sec", time.time()),
            )
            loaded_epoch = int(load_metadata.get("epoch", max(0, start_epoch - 1)))
            loaded_step = parse_step_from_checkpoint_id(load_metadata.get("checkpoint_id"))
            rollback_epochs = max(
                0.0,
                float(pending_recovery_context.get("fault_epoch", loaded_epoch) - loaded_epoch),
            )
            rollback_steps = max(
                0,
                int(pending_recovery_context.get("fault_global_step", 0) - loaded_step),
            )
            metrics.record_resume_success(
                resumed_from_epoch=start_epoch,
                checkpoint_id=load_metadata.get("checkpoint_id"),
                checkpoint_path=load_metadata.get("path"),
                load_source=load_source,
                time_to_resume_sec=time_to_resume_sec,
                rollback_epochs=rollback_epochs,
                rollback_steps=rollback_steps,
                lost_work_sec=0.0,
            )
            metrics.increment_detected_fault()
            metrics.record_recovery(success=True, recovery_time_sec=time_to_resume_sec)
            clear_recovery_context(args.checkpoint_dir)
            pending_recovery_context = None
        else:
            metrics.record_resume_success(
                resumed_from_epoch=start_epoch,
                checkpoint_id=load_metadata.get("checkpoint_id"),
                checkpoint_path=load_metadata.get("path"),
                load_source=load_source,
            )
        log_on_main(
            f"Resuming from epoch {start_epoch} (best_val_acc: {best_val_acc:.2f}%)"
        )
        if fault_cfg.enabled:
            log_on_main(
                "Checkpoint fault injection enabled by default | "
                f"type={fault_cfg.fault_type.value}, location={fault_cfg.fault_location}, "
                f"bit_flips={fault_cfg.num_bit_flips}, target_processes={fault_cfg.num_processes}/{world_size}"
            )

    # -------------------------
    # Training Loop
    # -------------------------
    log_on_main("\n" + "=" * 70)
    log_on_main("Starting Training")
    log_on_main("=" * 70)

    # best_val_acc already initialized above (0.0 for fresh start, or loaded from checkpoint on resume)

    global_step = start_epoch * max(1, len(train_loader))

    for epoch in range(start_epoch, args.epochs):
        try:
            # Record epoch start for metrics
            metrics.start_epoch(epoch)
            epoch_start = time.time()

            # Training
            log_on_main(f"\nEpoch {epoch + 1}/{args.epochs}")
            log_on_main("-" * 40)

            def checkpoint_callback(_batch_idx: int, batch_loss: float):
                nonlocal global_step
                nonlocal calibrated_checkpoint_cost_sec
                nonlocal checkpoint_cost_calibration_count
                global_step += 1
                metrics.mark_progress(epoch=epoch, global_step=global_step)

                if convergence_scheduler is None:
                    return

                now = time.time()
                convergence_scheduler.observe(batch_loss, now=now)
                
                # Update convergence metrics
                metrics.update_convergence_metrics(
                    current_interval_sec=convergence_scheduler.current_interval_seconds,
                    theta1=convergence_scheduler._last_fit.theta1 if convergence_scheduler._last_fit else None,
                    theta2=convergence_scheduler._last_fit.theta2 if convergence_scheduler._last_fit else None,
                )
                
                should_checkpoint_now = convergence_scheduler.should_checkpoint(now=now)

                # Keep checkpoint/save barrier flow consistent across ranks.
                # In DDP, a rank-local decision can deadlock if some ranks enter
                # checkpoint_manager.save() (which contains dist.barrier()) and others do not.
                if is_distributed_initialized():
                    checkpoint_flag = torch.tensor(
                        [1 if should_checkpoint_now else 0],
                        device=device,
                        dtype=torch.int32,
                    )
                    if not is_main_process():
                        checkpoint_flag.zero_()
                    torch.distributed.broadcast(checkpoint_flag, src=0)
                    should_checkpoint_now = bool(checkpoint_flag.item())

                if should_checkpoint_now:
                    checkpoint_id = f"step_{global_step:012d}"
                    ckpt_start = time.time()
                    checkpoint_manager.save(
                        model,
                        optimizer,
                        epoch + 1,
                        batch_loss,
                        checkpoint_id=checkpoint_id,
                    )
                    ckpt_time = time.time() - ckpt_start
                    
                    # Record checkpoint metrics
                    try:
                        ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_{checkpoint_id}.pt")
                        ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024) if os.path.exists(ckpt_path) else 0.0
                    except:
                        ckpt_size_mb = 0.0
                    
                    metrics.record_batch_checkpoint(batch_id=global_step, checkpoint_size_mb=ckpt_size_mb, save_time_sec=ckpt_time)
                    metrics.increment_convergence_checkpoint()
                    sync_hash_ring_metrics()

                    if args.auto_calibrate_checkpoint_cost:
                        if checkpoint_cost_calibration_count == 0:
                            calibrated_checkpoint_cost_sec = ckpt_time
                        else:
                            alpha = args.checkpoint_cost_ema_alpha
                            calibrated_checkpoint_cost_sec = (
                                alpha * ckpt_time
                                + (1.0 - alpha) * calibrated_checkpoint_cost_sec
                            )

                        checkpoint_cost_calibration_count += 1
                        if checkpoint_cost_calibration_count <= args.checkpoint_cost_warmup_saves:
                            convergence_scheduler.update_checkpoint_cost_seconds(
                                calibrated_checkpoint_cost_sec
                            )
                            log_on_main(
                                "  [Convergence CkptCost Calib] "
                                f"sample={checkpoint_cost_calibration_count}/{args.checkpoint_cost_warmup_saves} "
                                f"measured={ckpt_time:.3f}s "
                                f"calibrated={calibrated_checkpoint_cost_sec:.3f}s"
                            )

                        if checkpoint_cost_calibration_count == args.checkpoint_cost_warmup_saves:
                            log_on_main(
                                "  [Convergence CkptCost Calib] "
                                f"final checkpoint_cost_sec={calibrated_checkpoint_cost_sec:.3f}s"
                            )

                    maybe_inject_runtime_fault(checkpoint_id)
                    
                    convergence_scheduler.mark_checkpoint(now=now)
                    log_on_main(
                        "  [Convergence Ckpt] "
                        f"{checkpoint_id} | next_interval~{convergence_scheduler.current_interval_seconds:.1f}s"
                    )

            train_loss, train_correct, train_total = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                epoch,
                train_sampler,
                video_mode=args.video_mode or args.fast_video_mode,
                checkpoint_callback=checkpoint_callback,
            )

            # Synchronize after training
            if is_distributed_initialized():
                barrier()

            # Validation
            val_loss, val_correct, val_total, val_acc = evaluate(
                model,
                val_loader_simple,
                device,
                video_mode=args.video_mode or args.fast_video_mode,
            )

            # Update learning rate
            scheduler.step()

            # Aggregate metrics
            if is_distributed_initialized():
                train_loss, train_acc = reduce_metrics(
                    train_loss, train_correct, train_total, world_size
                )
                val_loss, val_acc = reduce_metrics(
                    val_loss, val_correct, val_total, world_size
                )
            else:
                train_acc = (
                    100.0 * train_correct / train_total if train_total > 0 else 0.0
                )

            epoch_time = time.time() - epoch_start

            # Record epoch metrics
            metrics.end_epoch(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc
            )
            metrics.mark_progress(epoch=epoch + 1, global_step=global_step)

            # Log results (main process only)
            log_on_main(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            log_on_main(
                f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}% | Time: {epoch_time:.1f}s"
            )
            log_on_main(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

            # Save checkpoint
            is_best = val_acc > best_val_acc

            if use_convergence:
                if is_best:
                    best_checkpoint_id = f"best_epoch_{epoch + 1:04d}"
                    ckpt_start = time.time()
                    checkpoint_manager.save(
                        model,
                        optimizer,
                        epoch + 1,
                        val_loss,
                        metric=val_acc,
                        checkpoint_id=best_checkpoint_id,
                    )
                    ckpt_time = time.time() - ckpt_start
                    
                    # Record checkpoint metrics
                    try:
                        ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_{best_checkpoint_id}.pt")
                        ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024) if os.path.exists(ckpt_path) else 0.0
                    except:
                        ckpt_size_mb = 0.0
                    metrics.record_checkpoint(checkpoint_size_mb=ckpt_size_mb, save_time_sec=ckpt_time, checkpoint_id=best_checkpoint_id)
                    sync_hash_ring_metrics()

                    maybe_inject_runtime_fault(best_checkpoint_id)
                    
                    best_val_acc = val_acc
                    log_on_main(
                        f"  ✓ New best model! Val Acc: {val_acc:.2f}% ({best_checkpoint_id})"
                    )
            else:
                should_save = (epoch + 1) % args.checkpoint_interval == 0
                if should_save or is_best:
                    ckpt_start = time.time()
                    checkpoint_manager.save(
                        model, optimizer, epoch + 1, val_loss, metric=val_acc
                    )
                    ckpt_time = time.time() - ckpt_start
                    
                    # Record checkpoint metrics
                    try:
                        ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                        ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024) if os.path.exists(ckpt_path) else 0.0
                    except:
                        ckpt_size_mb = 0.0
                    metrics.record_checkpoint(checkpoint_size_mb=ckpt_size_mb, save_time_sec=ckpt_time)
                    sync_hash_ring_metrics()

                    maybe_inject_runtime_fault(f"epoch_{epoch + 1:04d}")

                    if is_best:
                        best_val_acc = val_acc
                        log_on_main(f"  ✓ New best model! Val Acc: {val_acc:.2f}%")

        except Exception as e:
            # Barrier sync first - prevents rank 0 from saving while others are still running
            barrier()
            metrics.record_failure_event(
                reason=str(e),
                epoch=epoch,
                global_step=global_step,
            )

            # Emergency checkpoint save with current epoch and val_loss
            checkpoint_manager.save(
                model, optimizer, epoch, val_loss if "val_loss" in dir() else 0.0
            )
            sync_hash_ring_metrics()

            if use_hash_ring:
                checkpoint_manager.stop_heartbeat_thread()

            # Log error
            log_on_main(f"Training failed at epoch {epoch}: {e}")
            log_on_main("Saving emergency checkpoint and exiting...")

            # Re-raise to trigger torchrun restart
            raise

    # -------------------------
    # Training Complete
    # -------------------------
    log_on_main("\n" + "=" * 70)
    log_on_main("Training Complete")
    log_on_main("=" * 70)
    log_on_main(f"Best validation accuracy: {best_val_acc:.2f}%")
    log_on_main(f"Checkpoints saved to: {args.checkpoint_dir}")

    if use_hash_ring:
        checkpoint_manager.stop_heartbeat_thread()
    sync_hash_ring_metrics()

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
        
        try:
            html_path = exporter.export_html_report(metrics)
            log_on_main(f"  ✓ HTML report: {html_path}")
        except Exception as e:
            log_on_main(f"  ✗ HTML export failed: {e}")
        
        try:
            md_path = exporter.export_markdown_report(metrics)
            log_on_main(f"  ✓ Markdown report: {md_path}")
        except Exception as e:
            log_on_main(f"  ✗ Markdown export failed: {e}")
        
        log_on_main(f"\nMetrics saved to: ./experiment_results/{metrics.experiment_name}/")

    if is_distributed_initialized():
        cleanup_distributed()

    return 0


if __name__ == "__main__":
    sys.exit(main())
