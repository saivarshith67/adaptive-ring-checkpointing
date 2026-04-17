# Adaptive Ring Checkpointing

## What This Is

A PyTorch-based training pipeline with adaptive checkpoint timing strategies for fault-tolerant deep learning training. The system optimizes checkpoint frequency based on training dynamics to minimize overhead while ensuring reliable recovery from failures. Currently adding fault injection and automatic recovery capabilities for multi-GPU training.

## Core Value

Enable reliable long-running deep learning training with minimal checkpoint overhead through adaptive ring-based checkpoint strategies.

## Requirements

### Validated

- ✓ PyTorch training pipeline with CIFAR-100 dataset — existing
- ✓ Configurable checkpoint timing strategies (fixed, adaptive, ring) — existing
- ✓ Model checkpoint save/load functionality — existing
- ✓ Fault injection for testing checkpoint reliability — existing

### Active

- [ ] Implement fault injection in multi-GPU to simulate node failures
- [ ] Detect GPU/node failures during training
- [ ] Automatically recover from previous checkpoint after failure
- [ ] Verify fault tolerance end-to-end

### Out of Scope

- Real-time deep fake detection inference — not part of this project
- Model architecture changes beyond data loading

## Current Milestone: v1.1 Fault Tolerance

**Goal:** Enable multi-GPU training to recover from node/GPU failures by injecting faults and recovering from checkpoints.

**Target features:**
- Fault injection mechanism to simulate GPU/node failures in multi-GPU training
- Automatic failure detection during training
- Automatic recovery from previous checkpoint after failure detected
- End-to-end fault tolerance verification

---

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
| Use Kaggle DFD dataset | User-specified target dataset | ✓ Complete |
| Multi-GPU via PyTorch DDP | Standard approach for multi-GPU training | ✓ Complete |
| Keep existing checkpoint strategies | Works regardless of dataset/model | ✓ Complete |
| Fault injection via process kill | Simulate real-world node failures | — Pending |
| Auto-recovery from checkpoint | Resume training after failure detected | — Pending |

---
*Last updated: 2026-04-17 after v1.1 milestone started*
