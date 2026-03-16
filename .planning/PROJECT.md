# Adaptive Ring Checkpointing

## What This Is

A PyTorch-based training pipeline with adaptive checkpoint timing strategies for fault-tolerant deep learning training. The system optimizes checkpoint frequency based on training dynamics to minimize overhead while ensuring reliable recovery from failures.

## Core Value

Enable reliable long-running deep learning training with minimal checkpoint overhead through adaptive ring-based checkpoint strategies.

## Requirements

### Validated

- ✓ PyTorch training pipeline with CIFAR-100 dataset — existing
- ✓ Configurable checkpoint timing strategies (fixed, adaptive, ring) — existing
- ✓ Model checkpoint save/load functionality — existing
- ✓ Fault injection for testing checkpoint reliability — existing

### Active

- [ ] Integrate Deep Fake Detection dataset (DFD Entire Original) from Kaggle
- [ ] Implement multi-GPU training support using DataParallel/DistributedDataParallel
- [ ] Adapt training pipeline for new dataset structure (face images)
- [ ] Verify multi-GPU training works correctly

### Out of Scope

- Real-time deep fake detection inference — not part of this project
- Model architecture changes beyond data loading

## Context

**Existing codebase:** Modular PyTorch training pipeline with:
- Configuration management via YAML
- Model factory (ResNet, VGG, etc.)
- Dataset loaders (CIFAR-100, custom ImageDataset)
- Checkpoint strategies: FixedInterval, AdaptiveInterval, RingCheckpoint
- Fault injection testing

**Target dataset:** Deep Fake Detection (DFD) Entire Original Dataset from Kaggle
- Contains original face images for deep fake detection training
- Requires different preprocessing than CIFAR-100

## Constraints

- **Tech Stack**: Python 3.10+, PyTorch 2.10+ — existing
- **Multi-GPU**: Must work with NVIDIA GPUs using DataParallel or DistributedDataParallel
- **Dataset**: Must handle Kaggle DFD dataset structure

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Use Kaggle DFD dataset | User-specified target dataset | — Pending |
| Multi-GPU via PyTorch DDP | Standard approach for multi-GPU training | — Pending |
| Keep existing checkpoint strategies | Works regardless of dataset/model | — Pending |

---
*Last updated: 2026-03-16 after project initialization*
