# Feature Landscape: Multi-GPU Training with Custom Dataset

**Domain:** PyTorch Deep Learning Training
**Researched:** 2026-03-16

## Overview

This document categorizes features needed to add multi-GPU training support (via DistributedDataParallel) and integrate the Deep Fake Detection (DFD) dataset from Kaggle. The existing codebase has a functional single-GPU training loop with checkpoint strategies; this analysis determines what features are required to scale to multiple GPUs with a custom face image dataset.

---

## Table Stakes

Features that **must be implemented** or multi-GPU training will fail or produce incorrect results. Without these, the system cannot function as a distributed training system.

| Feature | Why Required | Complexity | Dependencies |
|---------|--------------|------------|--------------|
| **DDP Model Wrapping** | PyTorch's `DistributedDataParallel` is the standard for multi-GPU training. Required for gradient synchronization across GPUs. | Medium | None (pure PyTorch) |
| **Process Group Initialization** | DDP requires `torch.distributed.init_process_group()` with NCCL backend to coordinate GPUs. | Low | DDP wrapping |
| **DistributedSampler** | Replaces standard sampler to partition data across GPUs. Without this, all GPUs process identical data. | Low | Process group init |
| **Rank-Aware Checkpointing** | Only rank 0 should save checkpoints to avoid file conflicts. Requires `model.module` access to get underlying model state. | Low | DDP wrapping |
| **Device Placement by Rank** | Each process must use `local_rank` to select its GPU. Without this, all processes compete for GPU 0. | Low | Process group init |
| **Metrics Aggregation** | Loss/accuracy must be gathered across ranks via `all_reduce` for correct logging. | Low | Process group init |
| **DataLoader Worker Seeding** | Each worker needs unique seed to avoid duplicate data loading across ranks. | Low | DistributedSampler |

---

## Differentiators

Features that **add competitive value** beyond basic functionality. These are optional but improve training efficiency, fault tolerance, or usability.

| Feature | Value Proposition | Complexity | Dependencies |
|---------|------------------|------------|--------------|
| **Face-Specific Preprocessing** | DFD dataset contains face images requiring different transforms (face alignment, normalization) vs CIFAR-100. Improves model accuracy. | Medium | Table stakes |
| **DataLoader Optimization** | `pin_memory=True`, `num_workers > 0`, `persistent_workers=True` for faster data pipeline. | Low | None |
| **Gradient Scaling for Mixed Precision** | `GradScaler` from AMP enables FP16 training, reducing memory and improving throughput on modern GPUs. | Medium | DDP wrapping |
| **Fault-Tolerant DDP with torchrun** | Uses `torchrun` with elastic discovery for automatic restarts on worker failure. | Medium | DDP wrapping |
| **Multi-GPU Learning Rate Scaling** | Linearly scales LR with batch size (or uses warmup) to maintain training dynamics. | Low | Table stakes |
| **Distributed Evaluation** | Only rank 0 runs evaluation to avoid redundant computation while maintaining accuracy. | Low | Table stakes |
| **Async Checkpoint Saving** | Saves checkpoints in background thread to avoid blocking training. | Medium | Rank-aware checkpointing |
| **Checkpoint Resharding Support** | Supports changing GPU count between runs (via `torch.distributed.checkpoint`). | High | Table stakes |

---

## Anti-Features

Features to **deliberately NOT build**. Building these would waste effort or introduce unnecessary complexity.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **DataParallel (DP) Support** | Legacy single-process approach with GIL bottleneck. Less efficient than DDP. | Use DDP (already recommended) |
| **Multi-Node Training** | Out of scope for current requirements. Adds significant complexity (SSH keys, network config). | Keep single-node multi-GPU |
| **FSDP (Fully Sharded Data Parallel)** | Designed for models too large for a single GPU. Adds complexity without benefit for typical ResNet sizes. | Use DDP instead |
| **Real-Time Inference Pipeline** | Project is about fault-tolerant training, not deployment. | Separate project for inference |
| **Model Architecture Changes** | The project focuses on checkpoint strategies, not model research. | Keep existing model factory |
| **Custom Communication Backends** | NCCL is optimal for GPU training. Gloo is for CPU/multi-node edge cases. | Use NCCL exclusively |

---

## Feature Dependencies

```
DDP Model Wrapping
    ├── Process Group Initialization
    │   ├── Device Placement by Rank
    │   ├── Rank-Aware Checkpointing
    │   └── Metrics Aggregation
    ├── DistributedSampler
    │   └── DataLoader Worker Seeding
    └── Multi-GPU Learning Rate Scaling

Face-Specific Preprocessing ──► DataLoader Optimization

Gradient Scaling for Mixed Precision ──► DDP Model Wrapping

Fault-Tolerant DDP with torchrun ──► DDP Model Wrapping

Async Checkpoint Saving ──► Rank-Aware Checkpointing

Distributed Evaluation ──► Metrics Aggregation
```

---

## MVP Recommendation

Prioritize in this order:

### Phase 1: Core Multi-GPU (Table Stakes)
1. **DDP Model Wrapping** — Wrap existing model in `DistributedDataParallel`
2. **Process Group Initialization** — Add distributed init with NCCL backend
3. **DistributedSampler** — Replace default sampler in DataLoader
4. **Device Placement** — Map ranks to GPUs correctly

### Phase 2: Checkpoint Integration (Table Stakes)
5. **Rank-Aware Checkpointing** — Save only from rank 0, load on all ranks
6. **Metrics Aggregation** — Proper loss/accuracy logging across GPUs

### Phase 3: Dataset Integration (Differentiators)
7. **Face-Specific Preprocessing** — Adapt transforms for DFD face images
8. **DataLoader Optimization** — Add pin_memory, workers, persistent_workers

### Phase 4: Advanced Features (Optional Differentiators)
9. Mixed precision training
10. Async checkpointing
11. Fault-tolerant torchrun

---

## Sources

- **PyTorch DDP Tutorial**: https://pytorch.org/tutorials/beginner/ddp_series_intro
- **Distributed Data Parallel Best Practices**: https://medium.com/@geronimo7/getting-started-with-pytorch-ddp-3211a2cacaa7
- **DataLoader Optimization**: https://thelinuxcode.com/how-i-use-pytorch-dataloader-for-fast-reliable-training-pipelines-in-2026/
- **Face Preprocessing**: https://medium.com/@tunamuna29/face-landmarks-detection-with-deep-learning-using-pytorch-692ae27e2fdc
- **Checkpointing in DDP**: https://discuss.pytorch.org/t/right-ways-to-serialize-and-load-ddp-model-checkpoints/122719
