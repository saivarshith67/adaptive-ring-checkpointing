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
- ✓ Multi-GPU training with DDP — v1.0
- ✓ DFD dataset integration with face detection — v1.0
- ✓ Exception handling with try/except wrapper — v1.1
- ✓ Rank-aware fault injection — v1.1
- ✓ Checkpoint recovery — v1.1
- ✓ End-to-end fault tolerance verification — v1.1

### Active

- [ ] Implement hash ring data structure with virtual nodes
- [ ] Implement checkpoint shard manager with shard lifecycle
- [ ] Implement timeout-based fault detector with gossip quorum
- [ ] Implement elastic recaching engine
- [ ] Implement recovery scheduling with epoch rollback

### Out of Scope

- Real-time deep fake detection inference — not part of this project
- Model architecture changes beyond data loading

## Current Milestone: v1.2 Hash Ring

**Goal:** Implement hash ring-based elastic recaching with consistent hashing for efficient checkpoint shard management and fault tolerance.

**Target features:**
- Hash ring data structure with virtual nodes (100 per physical GPU)
- Checkpoint shard manager with lifecycle (UNCACHED → CACHED → ORPHANED → RE-CACHED)
- Timeout-based fault detector with gossip quorum
- Elastic recaching engine (one central storage access per lost shard)
- Recovery scheduling with epoch rollback protocol

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
| Fault injection via process kill | Simulate real-world node failures | ✓ Complete |
| Auto-recovery from checkpoint | Resume training after failure detected | ✓ Complete |
| Hash ring with 100 virtual nodes | Paper's optimal for load balancing | — Pending |
| Gossip-based failure detection | Avoids single point of failure | — Pending |
| One PFS access per lost shard | Core optimization for recaching | — Pending |

## Constraints

- **Tech Stack**: Python 3.10+, PyTorch 2.10+ — existing
- **Multi-GPU**: 4x NVIDIA A16 per node, CUDA 13.0 — existing
- **Dataset**: Must handle Kaggle DFD dataset structure — existing
- **Cluster**: Single-machine multi-GPU (not multi-node) — existing

---
*Last updated: 2026-04-18 after v1.2 milestone started*
