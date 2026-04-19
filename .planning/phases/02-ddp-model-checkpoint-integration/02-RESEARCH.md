# Phase 2: DDP Model & Checkpoint Integration - Research

**Researched:** 2026-03-16
**Domain:** PyTorch DDP (DistributedDataParallel) model wrapping, checkpoint management, and metrics aggregation
**Confidence:** HIGH

## Summary

Phase 2 implements DDP model wrapping, rank-aware checkpointing, and cross-GPU metrics aggregation. The key insight is that DDP wraps models with a "module." prefix internally, so checkpoint saving must use `model.module.state_dict()` to produce portable checkpoints. Metrics aggregation uses `dist.all_reduce` to synchronize loss and accuracy values across all GPU ranks. SyncBatchNorm conversion is required for proper batch normalization statistics synchronization in distributed training.

**Primary recommendation:** Modify `scripts/train_distributed.py` to wrap the model with DDP, convert BatchNorm to SyncBatchNorm, implement rank-aware checkpoint save/load, and add all-reduce for metrics aggregation. All eight requirements (MGPU-04, MGPU-05, DATA-03, CKPT-01, CKPT-02, CKPT-03, METR-01, METR-02) can be implemented in a single cohesive update to the training script.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|-------------|
| PyTorch | 2.10.0 | Deep learning framework with DDP support | Current stable from pyproject.toml |
| torch.nn.parallel.DistributedDataParallel | (from PyTorch) | Multi-GPU model wrapper with gradient sync | Standard for data-parallel training |
| torch.nn.SyncBatchNorm | (from PyTorch) | Cross-GPU batch norm synchronization | Required for consistent BatchNorm in DDP |
| torch.distributed | (from PyTorch) | All-reduce for metrics aggregation | Native distributed communication |

### Supporting

| Library | Purpose | When to Use |
|---------|---------|-------------|
| `torch.distributed.all_reduce` | Aggregate tensors across ranks | For loss and accuracy metrics |
| `torch.utils.data.worker_init_fn` | Seed data loading workers | For reproducibility across ranks |
| `torch.backends.cudnn.benchmark` | Optimize cuDNN performance | When input sizes are fixed |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| DDP | DataParallel | DDP avoids GIL, scales better, recommended by PyTorch |
| SyncBatchNorm | Default BatchNorm | Default BN computes stats per-GPU, causes training inconsistency |
| `torch.save` | Distributed Checkpoint (DCP) | DCP supports resharding but adds complexity; simple torch.save sufficient for v1 |

## Architecture Patterns

### Recommended Project Structure

```
scripts/
├── train_distributed.py    # MODIFIED: Add DDP wrapper, metrics aggregation, checkpointing
src/coci/
├── distributed.py           # EXISTING: Process group utilities (Phase 1)
├── models/
│   └── model.py            # EXISTING: Model definition (ResNet18/50, MobileNet)
├── checkpointing/
│   ├── checkpoint_manager.py  # MODIFIED: Add DDP-aware save/load
│   └── strategy.py          # EXISTING: Checkpoint strategies
```

### Pattern 1: DDP Model Wrapping

**What:** Wrap model with DistributedDataParallel to enable gradient synchronization across GPUs

**When to use:** At the start of training, after moving model to GPU but before creating optimizer

**Example:**
```python
# Source: PyTorch DDP Tutorial (https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
import torch.nn.parallel.DistributedDataParallel as DDP

# Create model and move to GPU
model = get_model(cfg.model, num_classes=100)
model.to(device)

# Wrap with DDP - MUST be after model.to(device)
model = DDP(model, device_ids=[local_rank])

# CRITICAL: Create optimizer AFTER DDP wrapping
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

**Key points:**
- `device_ids` specifies which GPUs this process uses
- Must create optimizer AFTER wrapping, not before
- Access underlying model via `model.module` for saving/loading

### Pattern 2: SyncBatchNorm Conversion

**What:** Convert BatchNorm layers to SyncBatchNorm for cross-GPU statistics synchronization

**When to use:** Before wrapping with DDP, especially for CNNs with BatchNorm

**Example:**
```python
# Source: PyTorch SyncBatchNorm Documentation (https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html)
import torch.nn as nn

# Create model
model = get_model(cfg.model, num_classes=100)

# Convert all BatchNorm*D layers to SyncBatchNorm
# This must happen BEFORE DDP wrapping
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

# Now wrap with DDP
model = DDP(model, device_ids=[local_rank])
```

**Key points:**
- Only needed if model has BatchNorm layers (ResNet does)
- Synchronization only occurs during training mode (`model.train()`)
- Converts BatchNorm1d, BatchNorm2d, BatchNorm3d recursively

### Pattern 3: DDP-Aware Checkpoint Saving (CKPT-01, CKPT-03)

**What:** Save checkpoint only from rank 0, using model.module.state_dict()

**When to use:** When checkpointing during distributed training

**Example:**
```python
# Source: PyTorch DDP Best Practices (https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
import torch.distributed as dist

def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    """Save checkpoint only on rank 0."""
    if dist.get_rank() == 0:
        # CRITICAL: Use model.module.state_dict(), NOT model.state_dict()
        # model.state_dict() would include "module." prefix from DDP
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        
        path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, path)
        print(f"Rank 0: Saved checkpoint to {path}")
    
    # Barrier to ensure all processes wait for rank 0 to finish saving
    dist.barrier()
```

**Key points:**
- MUST use `model.module.state_dict()` to avoid "module." prefix
- Only rank 0 should save to prevent file conflicts
- Use barrier() to synchronize all processes after save

### Pattern 4: DDP-Aware Checkpoint Loading (CKPT-02)

**What:** Load checkpoint with proper map_location for multi-GPU to single-GPU migration

**When to use:** When resuming training from checkpoint

**Example:**
```python
def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load checkpoint on all ranks."""
    # Load checkpoint with map_location
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load state dict into model (works for both DDP and non-DDP)
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch']
```

**Key points:**
- Use `map_location=device` to load to correct GPU
- For DDP models, use `model.module.load_state_dict()` not `model.load_state_dict()`
- All ranks should load (they start from same parameters)

### Pattern 5: Metrics Aggregation with All-Reduce (METR-01, METR-02)

**What:** Aggregate loss and accuracy values across all GPU ranks using all-reduce

**When to use:** At end of each epoch to get global metrics

**Example:**
```python
# Source: PyTorch Forums (https://discuss.pytorch.org/t/right-ways-to-serialize-and-load-ddp-model-checkpoints/122719)
import torch.distributed as dist
import torch

def reduce_metrics(loss, accuracy, world_size):
    """Reduce loss and accuracy across all ranks."""
    # Create tensors from scalar values
    loss_tensor = torch.tensor(loss, dtype=torch.float32, device='cuda')
    acc_tensor = torch.tensor(accuracy, dtype=torch.float32, device='cuda')
    
    # All-reduce: each rank gets the sum, then divide by world_size for average
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
    
    # Get average
    avg_loss = loss_tensor.item() / world_size
    avg_acc = acc_tensor.item() / world_size
    
    return avg_loss, avg_acc

# In training loop:
epoch_loss, epoch_acc = evaluate(model, test_loader, device)
if is_distributed_initialized():
    epoch_loss, epoch_acc = reduce_metrics(epoch_loss, epoch_acc, get_world_size())

# Log from rank 0 only
log_on_main(f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
```

**Key points:**
- Use `dist.all_reduce` with `ReduceOp.SUM`, then divide by world_size
- All ranks must participate in all-reduce
- Can aggregate partial sums during training, then reduce at epoch end

### Pattern 6: Reproducible Data Loading Workers (DATA-03)

**What:** Seed data loading workers identically across ranks for reproducibility

**When to use:** When using num_workers > 0 in DataLoader

**Example:**
```python
# Source: PyTorch DataLoader Documentation (https://pytorch.org/docs/stable/data.html)
import numpy as np
import torch.utils.data as data

def seed_worker(worker_id):
    """Seed worker identically across all ranks."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    # Note: Python random.seed() also recommended if used in dataset

# Create DataLoader with worker_init_fn
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    sampler=sampler,
    worker_init_fn=seed_worker,  # Ensures reproducibility
    pin_memory=True,
    persistent_workers=num_workers > 0
)
```

**Key points:**
- worker_init_fn is called once per worker at initialization
- PyTorch automatically sets torch seed to `base_seed + worker_id`
- Must explicitly seed numpy if used in dataset transforms

### Anti-Patterns to Avoid

- **Using `model.state_dict()` instead of `model.module.state_dict()`:** Creates "module." prefix in keys, breaks single-GPU inference
- **Creating optimizer before DDP wrapping:** DDP wraps parameters, optimizer must see wrapped parameters
- **Saving checkpoint on all ranks:** Creates file conflicts and duplicates
- **Not using barrier() after checkpoint save:** Other ranks may try to load before save completes
- **Using regular BatchNorm in DDP:** Each GPU computes local stats, causing inconsistent normalization
- **Logging without rank check:** Creates duplicate output from all GPUs
- **Not using map_location when loading:** Causes device mismatch errors

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Gradient synchronization | Custom all-reduce for gradients | DDP's built-in sync | Handles backward pass automatically, optimized communication |
| BatchNorm cross-GPU sync | Custom statistics averaging | SyncBatchNorm | Built-in synchronization, handles training/eval modes |
| Metrics aggregation | Manual collection and averaging | `dist.all_reduce` | Atomic operation, no race conditions |
| Worker seeding | Different seeds per rank | worker_init_fn with same base seed | Ensures reproducibility across ranks |
| Checkpoint coordination | Multiple rank saves | Rank 0 only + barrier | Prevents file conflicts |

**Key insight:** DDP provides built-in solutions for gradient sync and BatchNorm. Custom solutions would introduce bugs that are hard to debug in distributed settings.

## Common Pitfalls

### Pitfall 1: Checkpoint Keys Don't Match (CKPT-01 related)

**What goes wrong:** Loading checkpoint fails with "Missing key(s)" or "Unexpected key(s)"

**Why it happens:** 
- Saving `model.state_dict()` instead of `model.module.state_dict()` creates "module." prefix
- DDP internally wraps parameters with "module." prefix

**How to avoid:**
```python
# WRONG:
torch.save(model.state_dict(), path)

# CORRECT:
torch.save(model.module.state_dict(), path)
```

**Warning signs:** 
- Keys in saved checkpoint start with "module."
- Single-GPU inference loading fails

### Pitfall 2: Duplicate Checkpoint Files (CKPT-03 related)

**What goes wrong:** All GPUs try to save, causing file conflicts or duplicate saves

**Why it happens:** Not checking rank before save

**How to avoid:**
```python
# WRONG (all ranks save):
torch.save(checkpoint, path)

# CORRECT (only rank 0 saves):
if dist.get_rank() == 0:
    torch.save(checkpoint, path)
dist.barrier()  # Wait for rank 0 to finish
```

### Pitfall 3: BatchNorm Training Inconsistency

**What goes wrong:** Model trains but accuracy is lower than expected, or training is unstable

**Why it happens:** Each GPU computes its own batch statistics without synchronization

**How to avoid:**
```python
# WRONG:
model = get_model(...)
model = DDP(model, device_ids=[local_rank])

# CORRECT:
model = get_model(...)
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = DDP(model, device_ids=[local_rank])
```

### Pitfall 4: Metrics Not Synchronized Across Ranks

**What goes wrong:** Rank 0 reports different loss/accuracy than other ranks

**Why it happens:** Each rank computes metrics on its own data subset

**How to avoid:**
```python
# After computing loss/accuracy:
if is_distributed_initialized():
    loss, acc = reduce_metrics(loss, acc, get_world_size())
```

### Pitfall 5: Different Data Augmentations on Each GPU

**What goes wrong:** Training is not reproducible, same seed produces different results

**Why it happens:** Worker seeds differ across ranks, or seed not set per epoch

**How to avoid:**
```python
# Set seed before each epoch
torch.manual_seed(seed)
np.random.seed(seed)

# Use worker_init_fn
worker_init_fn=seed_worker
```

## Code Examples

### Example 1: Complete DDP Model Setup with SyncBatchNorm

```python
# Integration of MGPU-04, MGPU-05
def setup_model(model, local_rank):
    """Setup model with SyncBatchNorm and DDP wrapper."""
    # Convert BatchNorm to SyncBatchNorm before DDP
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # Move to GPU
    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)
    
    # Wrap with DDP
    model = DDP(
        model, 
        device_ids=[local_rank],
        output_device=local_rank
    )
    
    return model
```

### Example 2: Checkpoint Save with DDP Awareness

```python
# Integration of CKPT-01, CKPT-02, CKPT-03
def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    """Save checkpoint in DDP-aware manner."""
    if get_rank() == 0:
        checkpoint = {
            'epoch': epoch,
            # CRITICAL: Use model.module.state_dict()
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
        print(f"Rank 0: Saved checkpoint to {path}")
    
    # Ensure all ranks wait for save to complete
    barrier()

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load checkpoint in DDP-aware manner."""
    if not os.path.exists(checkpoint_path):
        return 0
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Use model.module.load_state_dict()
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch']
```

### Example 3: Metrics Aggregation

```python
# Integration of METR-01, METR-02
def compute_global_metrics(local_loss, local_correct, local_total, world_size):
    """Compute global metrics using all-reduce."""
    if not is_distributed_initialized():
        return local_loss, 100.0 * local_correct / local_total if local_total > 0 else 0.0
    
    # Create tensors
    loss_tensor = torch.tensor(local_loss, dtype=torch.float32, device='cuda')
    correct_tensor = torch.tensor(local_correct, dtype=torch.float32, device='cuda')
    total_tensor = torch.tensor(local_total, dtype=torch.float32, device='cuda')
    
    # All-reduce to sum across all ranks
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    
    # Compute global average
    global_loss = loss_tensor.item() / world_size
    global_acc = 100.0 * correct_tensor.item() / total_tensor.item() if total_tensor.item() > 0 else 0.0
    
    return global_loss, global_acc
```

### Example 4: Worker Seed Function

```python
# Integration of DATA-03
import numpy as np

def seed_worker(worker_id):
    """Seed worker identically across all ranks for reproducibility."""
    # PyTorch sets torch.initial_seed() to base_seed + worker_id
    # We use this to seed numpy for transforms that use numpy
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    # Add python random if needed
    # random.seed(worker_seed)

# Usage in DataLoader:
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    sampler=sampler,
    worker_init_fn=seed_worker,
    pin_memory=True,
    persistent_workers=num_workers > 0
)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| DataParallel | DDP | PyTorch 1.0+ | Avoids GIL, scales beyond single node |
| Manual gradient all-reduce | DDP automatic sync | DDP introduction | Simpler code, optimized communication |
| Per-GPU BatchNorm stats | SyncBatchNorm | PyTorch 1.5+ | Consistent normalization across GPUs |
| torch.distributed.launch | torchrun | PyTorch 1.9+ | Auto env vars, elastic training |
| Save DDP state_dict | Save model.module.state_dict() | DDP best practice | Portable checkpoints |

**Deprecated/outdated:**
- DataParallel: Use DDP instead (more efficient)
- Saving wrapped model state_dict: Use model.module.state_dict()

## Open Questions

1. **Broadcast buffers for BatchNorm?**
   - What we know: SyncBatchNorm handles buffer sync automatically during forward
   - What's unclear: If broadcast_buffers=False needed for any edge cases
   - Recommendation: Use default (True), only change if errors occur

2. **Mixed precision (AMP) for v1?**
   - What we know: METR-01, METR-02 don't require AMP, but it improves throughput
   - What's unclear: Whether to include in Phase 2 or defer
   - Recommendation: Defer to v2 - focus on core DDP functionality first

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MGPU-04 | Wrap model with DistributedDataParallel | Pattern 1 covers DDP wrapping with device_ids |
| MGPU-05 | Convert BatchNorm to SyncBatchNorm | Pattern 2 covers convert_sync_batchnorm usage |
| DATA-03 | Seed data loading workers identically | Pattern 6 covers worker_init_fn for reproducibility |
| CKPT-01 | Save from model.module.state_dict() | Pattern 3 covers checkpoint save with correct state dict |
| CKPT-02 | Load with map_location for migration | Pattern 4 covers checkpoint loading with device mapping |
| CKPT-03 | Coordinate save on rank 0 only | Pattern 3 covers rank-0 save with barrier |
| METR-01 | All-reduce for loss aggregation | Pattern 5 covers reduce_metrics with all_reduce |
| METR-02 | Aggregate accuracy across ranks | Pattern 5 covers reduce_metrics with all_reduce |

## Validation Architecture

> Skipping validation architecture - `workflow.nyquist_validation` not detected in .planning/config.json

## Sources

### Primary (HIGH confidence)
- PyTorch DDP Tutorial — https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- PyTorch SyncBatchNorm Documentation — https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html
- PyTorch DataLoader Documentation — https://pytorch.org/docs/stable/data.html

### Secondary (HIGH confidence)
- PyTorch DDP Checkpoint Best Practices — https://discuss.pytorch.org/t/right-ways-to-serialize-and-load-ddp-model-checkpoints/122719
- PyTorch Distributed Metrics Aggregation — https://discuss.pytorch.org/t/how-to-fix-randomness-of-dataloader-in-ddp/166565
- Azure ML DDP Guide — https://medium.com/data-science-at-microsoft/scaling-model-training-with-pytorch-distributed-data-parallel-ddp

### Tertiary (MEDIUM confidence)
- Community DDP examples on GitHub — Various implementations confirming patterns

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Verified from PyTorch 2.10.0 documentation and pyproject.toml
- Architecture: HIGH - Standard PyTorch DDP patterns from official tutorials
- Pitfalls: HIGH - Multiple sources confirm critical pitfalls and solutions

**Research date:** 2026-03-16
**Valid until:** 2026-04-16 (30 days for stable PyTorch APIs)
