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
import os
import sys
import time
import json
from pathlib import Path
from typing import List, Optional, Tuple

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
    model, train_loader, optimizer, device, epoch, sampler=None, video_mode=False
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
def main():
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

    args = parser.parse_args()

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
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        is_ddp_wrapped=True,
    )

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        start_epoch = checkpoint_manager.load_latest(model, optimizer, device)
        log_on_main(f"Resuming from epoch {start_epoch}")

    # -------------------------
    # Training Loop
    # -------------------------
    log_on_main("\n" + "=" * 70)
    log_on_main("Starting Training")
    log_on_main("=" * 70)

    best_val_acc = 0.0

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Training
        log_on_main(f"\nEpoch {epoch + 1}/{args.epochs}")
        log_on_main("-" * 40)

        train_loss, train_correct, train_total = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            train_sampler,
            video_mode=args.video_mode or args.fast_video_mode,
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
            train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0.0

        epoch_time = time.time() - epoch_start

        # Log results (main process only)
        log_on_main(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        log_on_main(
            f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}% | Time: {epoch_time:.1f}s"
        )
        log_on_main(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint
        should_save = (epoch + 1) % args.checkpoint_interval == 0
        is_best = val_acc > best_val_acc

        if should_save or is_best:
            checkpoint_manager.save(
                model, optimizer, epoch + 1, val_loss, metric=val_acc
            )

            if is_best:
                best_val_acc = val_acc
                log_on_main(f"  ✓ New best model! Val Acc: {val_acc:.2f}%")

    # -------------------------
    # Training Complete
    # -------------------------
    log_on_main("\n" + "=" * 70)
    log_on_main("Training Complete")
    log_on_main("=" * 70)
    log_on_main(f"Best validation accuracy: {best_val_acc:.2f}%")
    log_on_main(f"Checkpoints saved to: {args.checkpoint_dir}")

    if is_distributed_initialized():
        cleanup_distributed()

    return 0


if __name__ == "__main__":
    sys.exit(main())
