# Architecture

**Analysis Date:** 2026-04-18

## Pattern Overview

**Overall:** Modular training framework with distributed checkpointing and fault tolerance

**Key Characteristics:**
- Multi-GPU distributed training via PyTorch DDP with NCCL backend
- Adaptive checkpoint strategies (fixed interval, Young-Daly formula, epoch-based)
- Ring-based sharding for distributed cache management with fault detection
- Fault injection for testing fault tolerance

## Layers

**Training Layer:**
- Purpose: Orchestrate training loop, evaluation, and fault handling
- Location: `scripts/train.py`, `scripts/train_distributed.py`
- Contains: Training loops, evaluation, experiment logging
- Depends on: Model, Data, Checkpointing, Fault Injection layers
- Used by: Main entry points

**Model Layer:**
- Purpose: Neural network architectures and model creation
- Location: `src/coci/models/model.py`
- Contains: `get_model()`, `get_efficientnet_binary()`, `MultiFrameModel`
- Depends on: torch, torchvision.models
- Used by: Training scripts

**Data Ingestion Layer:**
- Purpose: Dataset loading and preprocessing
- Location: `src/coci/data_ingestor/`
- Contains: `dataset.py`, `cifar.py`, `faceforensics.py`
- Depends on: torch.utils.data, opencv-python
- Used by: Training scripts

**Checkpointing Layer:**
- Purpose: Model/optimizer state persistence and recovery
- Location: `src/coci/checkpointing/`
- Contains: `checkpoint_manager.py`, `strategy.py`
- Depends on: torch, strategy pattern
- Used by: Training scripts

**Distributed Layer:**
- Purpose: Multi-GPU coordination and process group management
- Location: `src/coci/distributed.py`
- Contains: rank detection, barrier, metrics aggregation
- Depends on: torch.distributed
- Used by: train_distributed.py, checkpoint_manager

**Hashing/Ring Layer:**
- Purpose: Distributed cache sharding and fault detection
- Location: `src/coci/hashing/`
- Contains: `hash_ring.py`, `fault_detector.py`, `recovery_scheduler.py`
- Depends on: hashlib, threading
- Used by: Cache management (not yet integrated into training)

**Fault Injection Layer:**
- Purpose: Simulate failures for fault tolerance testing
- Location: `src/coci/fault/fault_injector.py`
- Contains: Poisson-based failure injection
- Depends on: random, time, math
- Used by: train.py (via --inject_fault flag)

**Configuration Layer:**
- Purpose: Configuration loading and dataclass definition
- Location: `src/coci/config.py`
- Contains: Config dataclass, load_config()
- Depends on: yaml, dataclasses
- Used by: All scripts

## Data Flow

**Single-GPU Training:**
```
train.py → Config → Load dataset → Model → Training Loop
                                  ↓
                           CheckpointStrategy (should_checkpoint?)
                                  ↓
                           CheckpointManager.save()
```

**Distributed Training:**
```
train_distributed.py → setup_distributed() → DDP wrapper
                                          ↓
                           DistributedSampler (data partitioning)
                                          ↓
                           Training Loop with barrier()
                                          ↓
                           reduce_metrics() → all-reduce across GPUs
```

**Checkpoint Recovery:**
```
RuntimeError caught → train.py exits with code 1
→ Re-run train.py → CheckpointManager.load_latest()
→ Resume from saved epoch
```

**Ring-based Sharding:**
```
HashRing.add_node() → SHA256 position mapping
                     → ShardManager.register_shard()
                     → FaultDetector monitors heartbeats
                     → ElasticRecaching handles node failures
```

## Key Abstractions

**CheckpointStrategy (Abstract):**
- Purpose: Determine when to save checkpoints
- Examples: `FixedIntervalStrategy`, `YoungDalyStrategy`, `EpochStrategy`
- Pattern: Strategy pattern via CheckpointStrategyFactory

**HashRing:**
- Purpose: Consistent hashing for shard distribution
- Examples: `hash_ring.py` - HashRing, ShardManager classes
- Pattern: Virtual nodes for even distribution

**FaultDetector:**
- Purpose: Node health monitoring with suspicion quorum
- Examples: `fault_detector.py` - NodeStatus enum, ElasticRecaching
- Pattern: Phi accrual failure detector variant

## Entry Points

**Single-GPU:**
- Location: `scripts/train.py`
- Triggers: `python scripts/train.py [--mode {dev,server}] [--inject_fault]`
- Responsibilities: Config loading, dataset creation, model training, checkpointing, fault injection

**Distributed:**
- Location: `scripts/train_distributed.py`
- Triggers: `torchrun --nproc_per_node=N scripts/train_distributed.py`
- Responsibilities: Distributed setup, DDP wrapping, epoch synchronization, metrics aggregation

**Main Stub:**
- Location: `src/main.py`
- Triggers: `python -m src.main`
- Responsibilities: Placeholder (just prints hello)

## Error Handling

**Strategy:** Graceful degradation with checkpoint recovery

**Patterns:**
- RuntimeError catching in train.py with crash logging to `crash_experiment_log_*.jsonl`
- Distributed cleanup via `cleanup_distributed()` in finally/except blocks
- Checkpoint load returns (start_epoch, best_metric) to resume correctly
- Rank-aware checkpoint saving (only rank 0 saves) with barrier for sync

## Cross-Cutting Concerns

**Logging:** `log_on_main()` for rank-0-only output, experiment summary printing
**Validation:** Config loaded from YAML via `load_config()`, validated at dataclass level
**Authentication:** Not applicable (local training)

---

*Architecture analysis: 2026-04-18*