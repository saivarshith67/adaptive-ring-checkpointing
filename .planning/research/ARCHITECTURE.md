# Multi-GPU Training Architecture

**Domain:** Deep Learning Training Systems
**Researched:** 2026-03-16
**Confidence:** HIGH

## Executive Summary

PyTorch multi-GPU training is typically implemented using **DistributedDataParallel (DDP)**, which is the recommended approach over DataParallel due to its superior performance, multi-process architecture, and better gradient synchronization. DDP replicates the model on each GPU, processes different data batches, and synchronizes gradients using all-reduce operations.

Integration with existing checkpointing requires careful handling: **save from `model.module.state_dict()`** (unwrapped) and **load on all ranks** with proper device mapping.

---

## Recommended Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Multi-Process Entry Point                   │
│                  (torch.multiprocessing.spawn)                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   ┌─────────┐        ┌─────────┐       ┌─────────┐
   │ Rank 0  │        │ Rank 1  │  ...  │ Rank N  │
   │ GPU 0   │        │ GPU 1   │       │ GPU N   │
   └────┬────┘        └────┬────┘       └────┬────┘
        │                  │                  │
        ▼                  ▼                  ▼
   ┌─────────────────────────────────────────┐
   │        Process Group (NCCL/Gloo)        │
   │    Gradient Synchronization via         │
   │         All-Reduce Collective           │
   └─────────────────────────────────────────┘
```

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| **Process Group** | Initialize distributed communication (NCCL backend) | All ranks |
| **DistributedSampler** | Partition data across GPUs, ensure no overlap | DataLoader |
| **DDP Wrapper** | Replicate model, synchronize gradients | Process Group |
| **Checkpoint Manager** | Save/load state with DDP awareness | Rank 0 (save), All ranks (load) |
| **Metric Aggregator** | Collect loss/accuracy across ranks | All ranks → Rank 0 |

### Data Flow

```
Training Iteration:
1. DistributedSampler → Each rank gets distinct batch slice
2. Forward Pass → model(batch) on each GPU
3. Loss Computation → local loss per GPU
4. Backward Pass → local gradients computed
5. DDP All-Reduce → gradients synchronized across all GPUs
6. Optimizer Step → all GPUs apply same updated weights
7. Checkpoint Decision → strategy.should_checkpoint()
8. If checkpoint: Rank 0 saves model.module.state_dict()
```

---

## Integration with Existing Architecture

### What Changes (Single → Multi-GPU)

| Existing Component | Multi-GPU Equivalent | Change Type |
|-------------------|---------------------|-------------|
| `DataLoader` | `DataLoader` + `DistributedSampler` | Modified |
| `model.to(device)` | `model.to(rank)` + `DDP(model)` | Modified |
| `checkpoint_manager.save()` | Save `model.module.state_dict()` | Modified |
| `checkpoint_manager.load()` | Load on all ranks with `map_location` | Modified |
| Single-process entry | `mp.spawn()` or `torchrun` entry | New |

### What Stays the Same

- **CheckpointStrategy**: Works identically (per-batch timing decisions)
- **CheckpointManager**: Same interface, just DDP-aware internals
- **Model architecture**: No changes to model definition
- **Training loop logic**: Forward → Loss → Backward → Step pattern preserved

---

## Build Order (Dependencies)

### Phase 1: Infrastructure Setup
```
1. Add distributed entry point (torchrun or mp.spawn)
   ↓
2. Add process group initialization (init_process_group)
   ↓
3. Add rank/local_rank handling in config
```

**Dependencies:** No dependencies on existing code. Pure addition.

### Phase 2: Data Layer
```
1. Add DistributedSampler to DataLoader
   ↓
2. Add epoch synchronization (sampler.set_epoch)
```

**Dependencies:** Phase 1 (need rank to partition correctly)

### Phase 3: Model Layer
```
1. Wrap model with DDP: ddp_model = DDP(model, device_ids=[local_rank])
   ↓
2. Use ddp_model in training loop (not raw model)
```

**Dependencies:** Phase 1 (need process group), Phase 2 (need data)

### Phase 4: Checkpoint Integration
```
1. Modify save: checkpoint_manager.save(ddp_model.module, ...)
   ↓
2. Modify load: load on all ranks with map_location
   ↓
3. Ensure only rank 0 writes to disk
```

**Dependencies:** Phase 3 (need DDP wrapper to access .module)

### Phase 5: Metrics & Logging
```
1. Add all-reduce for loss aggregation
   ↓
2. Add accuracy aggregation across ranks
   ↓
3. Ensure rank 0 handles file I/O only
```

**Dependencies:** Phase 3 (need DDP for collective ops)

---

## Key Implementation Patterns

### Pattern 1: DDP Model Wrapping

```python
# Single-GPU (existing)
model = get_model(cfg.model, num_classes=100)
model.to(device)

# Multi-GPU (new)
model = get_model(cfg.model, num_classes=100)
model.to(local_rank)
model = DDP(model, device_ids=[local_rank])
```

### Pattern 2: Checkpoint Save (DDP-Aware)

```python
# WRONG - will save DDP internal state
torch.save(model.state_dict(), 'checkpoint.pth')

# CORRECT - unwrap to get original model
if rank == 0:  # Only rank 0 writes
    torch.save({
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, 'checkpoint.pth')
```

### Pattern 3: Checkpoint Load (DDP-Aware)

```python
# All ranks load (required for DDP)
checkpoint = torch.load('checkpoint.pth', map_location=f'cuda:{local_rank}')
model.module.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### Pattern 4: DistributedSampler Usage

```python
from torch.utils.data.distributed import DistributedSampler

train_sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True
)

train_loader = DataLoader(
    dataset,
    batch_size=cfg.batch_size,
    sampler=train_sampler,
    # Important: num_workers per GPU, not total
    num_workers=cfg.num_workers // world_size
)
```

### Pattern 5: Metric Aggregation

```python
def reduce_tensor(tensor):
    """All-reduce to get mean across all GPUs"""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

# Usage in training loop
loss = F.cross_entropy(outputs, labels)
loss = reduce_tensor(loss)  # Aggregate before logging
```

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Saving from Wrapped Model
**What:** `torch.save(model.state_dict(), ...)`
**Why:** Saves DDP internal buffers (`._comm_buffers`), not the actual model weights
**Instead:** Always use `model.module.state_dict()`

### Anti-Pattern 2: Saving from All Ranks
**What:** Every rank writes checkpoint to same file
**Why:** File corruption, race conditions
**Instead:** Use `if rank == 0:` guard for disk I/O

### Anti-Pattern 3: Loading Only on Rank 0
**What:** Loading checkpoint only on rank 0, then hoping synchronization works
**Why:** DDP requires all processes to have same initial state
**Instead:** All ranks load, or use broadcast from rank 0

### Anti-Pattern 4: Using Regular Sampler with DDP
**What:** Using `shuffle=True` instead of DistributedSampler
**Why:** All GPUs get same data, no parallelism
**Instead:** Use DistributedSampler with `shuffle` parameter

### Anti-Pattern 5: Missing num_workers Division
**What:** Using total num_workers from config directly
**Why:** Too many CPU workers, memory pressure
**Instead:** `num_workers // world_size`

---

## Scalability Considerations

| Concern | At 2 GPUs | At 8 GPUs | At 64+ GPUs |
|---------|-----------|------------|-------------|
| **Communication** | Minimal overhead | NCCL bandwidth critical | Network topology matters |
| **Data Loading** | 2x throughput | 8x throughput | May become bottleneck |
| **Checkpoint I/O** | Single file ok | Parallel save needed | Distributed checkpoint (DCP) |
| **Memory** | ~2x model | ~8x model | Consider FSDP |

---

## Integration with Existing Checkpoint Strategies

The existing checkpoint timing strategies (`FixedInterval`, `AdaptiveInterval`, `RingCheckpoint`) work **unchanged** with DDP because:

1. **Timing decisions** happen per-process at the same logical time
2. **Strategy state** is local but should be identical across ranks
3. **Save/load** is the only DDP-specific part

**Required adaptation:**
- `strategy.should_checkpoint()` called on all ranks (redundant but safe)
- Actual save happens only on rank 0
- On load, all ranks restore from same checkpoint → states identical

---

## Sources

- [PyTorch DDP Documentation](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) — **HIGH confidence**
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) — **HIGH confidence**
- [DDP Internal Design Notes](https://pytorch.org/docs/stable/notes/ddp.html) — **HIGH confidence**
- [Multi-GPU Training with DDP](https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html) — **HIGH confidence**
- [Distributed Checkpointing (DCP)](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html) — **HIGH confidence**
- [PyTorch Forums: DDP Checkpoint Best Practices](https://discuss.pytorch.org/t/right-ways-to-serialize-and-load-ddp-model-checkpoints/122719) — **MEDIUM confidence**
