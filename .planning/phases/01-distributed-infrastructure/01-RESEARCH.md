# Phase 1: Distributed Infrastructure - Research

**Researched:** 2026-03-16
**Domain:** PyTorch DDP (DistributedDataParallel) multi-GPU training infrastructure
**Confidence:** HIGH

## Summary

Phase 1 establishes the foundational multi-GPU training infrastructure using PyTorch's DistributedDataParallel (DDP). The phase addresses process group initialization, torchrun-based launching, rank-aware code paths, data distribution via DistributedSampler, epoch synchronization, and rank-0-only logging. This is standard PyTorch distributed training territory with well-established patterns.

**Primary recommendation:** Implement a modular `distributed.py` module that handles process group setup, provides utility functions for rank detection and rank-aware operations, and integrate DistributedSampler into the existing DataLoader. Use CIFAR-100 for validation (already available in project).

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.10.0 | Deep learning framework | Current stable (from pyproject.toml) |
| torchvision | 0.25.0 | CIFAR-100 dataset, transforms | Matches PyTorch version |
| NCCL backend | (from PyTorch) | GPU inter-process communication | Required for multi-GPU training |

### Supporting

| Library | Purpose | When to Use |
|---------|---------|-------------|
| `torch.distributed` | Process group initialization, all-reduce | All DDP training code |
| `torch.nn.parallel.DistributedDataParallel` | Model wrapping (Phase 2) | For actual distributed training |
| `torch.utils.data.DistributedSampler` | Data partitioning across GPUs | When using multiple GPUs |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| NCCL | Gloo (CPU) | Gloo too slow for GPU training; NCCL is standard |
| DDP | DataParallel | DDP avoids GIL, scales better, recommended by PyTorch |
| torchrun | `torch.multiprocessing.spawn` | torchrun handles env vars automatically, simpler |

## Architecture Patterns

### Recommended Project Structure

```
scripts/
├── train.py              # Main entry point (existing)
├── train_distributed.py  # NEW: torchrun-based distributed entry
src/coci/
├── distributed.py        # NEW: Process group setup & utilities
├── models/               # (existing)
├── data_ingestor/        # (existing)
├── checkpointing/        # (existing)
```

### Pattern 1: Process Group Initialization

**What:** Initialize NCCL process group and set local GPU device for each process

**When to use:** At the start of any distributed training script

**Example:**
```python
# Source: PyTorch DDP Tutorial (https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
import os
import torch
import torch.distributed as dist

def setup_distributed():
    """Initialize process group for distributed training."""
    # Get local rank from torchrun environment
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Initialize NCCL process group
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    
    # Set GPU device for this process
    torch.cuda.set_device(local_rank)
    
    return local_rank, rank, world_size
```

### Pattern 2: torchrun Launch Configuration

**What:** Command-line arguments for launching multi-GPU training

**When to use:** When launching training scripts with torchrun

**Example:**
```bash
# Single-node multi-GPU training
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    train_distributed.py

# With elastic training (fault-tolerant)
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    --max_restarts=3 \
    train_distributed.py
```

### Pattern 3: DistributedSampler Integration

**What:** Partition dataset across GPUs so each process gets unique batches

**When to use:** When creating DataLoader for distributed training

**Example:**
```python
# Source: PyTorch DDP Tutorial (adapted)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def get_distributed_dataloader(dataset, batch_size, num_workers, rank, world_size):
    """Create DataLoader with DistributedSampler."""
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42,  # Fixed seed for reproducibility
        drop_last=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=num_workers > 0
    )
    
    return dataloader, sampler
```

### Pattern 4: Epoch Synchronization

**What:** Call `sampler.set_epoch(epoch)` at the start of each epoch

**When to use:** At the beginning of each training epoch to ensure proper shuffling

**Example:**
```python
# Source: PyTorch DistributedSampler documentation
for epoch in range(num_epochs):
    # CRITICAL: Set epoch before starting the epoch
    sampler.set_epoch(epoch)
    
    for batch in train_loader:
        # Training code...
        pass
    
    # Synchronize at epoch end (optional but recommended)
    dist.barrier()
```

### Pattern 5: Rank-Aware Logging

**What:** Only rank 0 should print to console to avoid duplicate output

**When to use:** Any print/logging statement in training code

**Example:**
```python
# Source: PyTorch DDP best practices
def log_on_rank_0(message, *args, **kwargs):
    """Print only on rank 0."""
    if dist.get_rank() == 0:
        print(message, *args, **kwargs)

def is_main_process():
    """Check if current process is rank 0."""
    return dist.get_rank() == 0
```

### Pattern 6: Cleanup

**What:** Properly destroy process group at end of training

**When to use:** At script exit or when cleaning up resources

**Example:**
```python
def cleanup():
    """Clean up distributed process group."""
    dist.destroy_process_group()
```

### Anti-Patterns to Avoid

- **Using DataParallel instead of DDP:** DataParallel uses multi-threading (suffers from GIL), less efficient
- **Not calling `sampler.set_epoch(epoch)`:** Results in same data ordering every epoch
- **Printing without rank check:** Creates unreadable duplicate output
- **Using Gloo backend for GPU training:** Too slow; use NCCL
- **Not setting `LOCAL_RANK` for CUDA device:** Causes device mismatches

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Process group init | Custom initialization code | `dist.init_process_group()` with NCCL | Handles NCCL timeout, rendezvous, environment variables |
| Data partitioning | Custom sharding logic | `DistributedSampler` | Handles edge cases ( uneven dataset size), maintains reproducibility |
| Gradient sync | Custom all-reduce code | DDP's built-in sync | Optimized, handles backward pass automatically |
| Rank detection | Parse args manually | `os.environ["LOCAL_RANK"]` | torchrun sets automatically |

**Key insight:** DDP is a mature, battle-tested system. Custom solutions for process group, data sharding, or gradient synchronization introduce subtle bugs that are hard to debug.

## Common Pitfalls

### Pitfall 1: NCCL Initialization Timeout

**What goes wrong:** Training hangs at startup, never progresses

**Why it happens:** 
- NCCL can't establish peer-to-peer communication
- Firewall blocking ports
- NVLink/PCIe configuration issues

**How to avoid:**
```bash
# Common fix: disable NCCL optimizations that cause issues
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
```

**Warning signs:** 
- Process hangs after "init process group"
- `NCCL timeout` errors in logs

### Pitfall 2: Duplicate Logging Output

**What goes wrong:** Every GPU prints the same message, output is unreadable

**Why it happens:** Each process runs the same code including print statements

**How to avoid:** Wrap all prints with rank check:
```python
if dist.get_rank() == 0:
    print(f"Epoch {epoch}: Loss = {loss}")
```

### Pitfall 3: Same Data on Every GPU

**What goes wrong:** All GPUs process identical batches, no speedup

**Why it happens:** Using regular `RandomSampler` instead of `DistributedSampler`

**How to avoid:** Always pass `sampler` to DataLoader, not `shuffle=True`:
```python
# WRONG:
DataLoader(dataset, shuffle=True, ...)

# CORRECT:
sampler = DistributedSampler(dataset, ...)
DataLoader(dataset, sampler=sampler, ...)
```

### Pitfall 4: Stale Random State After Checkpoint Resume

**What goes wrong:** All GPUs use identical data batches after resuming from checkpoint

**Why it happens:** Not calling `sampler.set_epoch()` after checkpoint load

**How to avoid:** Always call `sampler.set_epoch(resume_epoch)` after loading checkpoint

## Code Examples

### Example 1: Minimal Distributed Training Entry Point

```python
# scripts/train_distributed.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def setup():
    """Initialize distributed training."""
    # Get environment variables set by torchrun
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # Initialize process group
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    return local_rank, rank, world_size

def cleanup():
    """Clean up distributed resources."""
    dist.destroy_process_group()

def main():
    local_rank, rank, world_size = setup()
    
    if rank == 0:
        print(f"Starting distributed training with {world_size} GPUs")
    
    # Your existing training code here...
    
    cleanup()

if __name__ == "__main__":
    main()
```

### Example 2: Wrapper Function for Distributed Setup

```python
# src/coci/distributed.py
"""Distributed training utilities."""
import os
import torch
import torch.distributed as dist


def is_distributed_initialized():
    """Check if distributed is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank():
    """Get current process rank."""
    if is_distributed_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    """Get total number of processes."""
    if is_distributed_initialized():
        return dist.get_world_size()
    return 1


def get_local_rank():
    """Get local rank (GPU index)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process():
    """Check if current process is rank 0."""
    return get_rank() == 0


def barrier():
    """Synchronize all processes."""
    if is_distributed_initialized():
        dist.barrier()


def setup_distributed(backend="nccl"):
    """Initialize process group."""
    if is_distributed_initialized():
        return
    
    local_rank = get_local_rank()
    rank = get_rank()
    world_size = get_world_size()
    
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(local_rank)


def cleanup_distributed():
    """Destroy process group."""
    if is_distributed_initialized():
        dist.destroy_process_group()


def log_on_main(*args, **kwargs):
    """Print only on main process."""
    if is_main_process():
        print(*args, **kwargs)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torch.distributed.launch` | `torchrun` | PyTorch 1.9+ (2020) | Auto env vars, elastic training support |
| DataParallel | DDP | PyTorch 1.0+ | Avoids GIL, scales beyond single node |
| Gloo backend for GPUs | NCCL backend | Standard | 5-10x faster GPU communication |
| Manual rank detection | `LOCAL_RANK` env var | torchrun | Simpler, less error-prone |

**Deprecated/outdated:**
- `torch.distributed.launch`: Replaced by torchrun (still works but not recommended)
- DataParallel: Use DDP instead (more efficient)

## Open Questions

1. **Multi-node training for v1?**
   - What we know: ROADMAP says single machine multi-GPU only for v1
   - What's unclear: If single machine has >8 GPUs, might need NVLink tuning
   - Recommendation: Start single-node, expand if needed

2. **NCCL timeout configuration?**
   - What we know: Default 30 min timeout may be too short for large models
   - What's unclear: Optimal timeout for development vs production
   - Recommendation: Use env var `NCCL_TIMEOUT=3600` for development

3. **Elastic training for fault recovery?**
   - What we know: torchrun supports `--max_restarts`
   - What's unclear: How it interacts with checkpoint strategies
   - Recommendation: Defer to v2 - focus on basic DDP first

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MGPU-01 | Initialize distributed process group | Pattern 1 covers NCCL init with proper env var handling |
| MGPU-02 | Add torchrun-based entry point | Pattern 2 covers torchrun launch configuration |
| MGPU-03 | Implement rank-aware code paths | Patterns 4-5 cover rank detection and rank-0-only logging |
| DATA-01 | Integrate DistributedSampler | Pattern 3 covers DistributedSampler integration |
| DATA-02 | Synchronize epoch count | Pattern 4 covers set_epoch synchronization |
| METR-03 | Ensure logging only on rank 0 | Pattern 5 covers rank-aware logging |

## Validation Architecture

> Skipping validation architecture - `workflow.nyquist_validation` not detected in .planning/config.json

## Sources

### Primary (HIGH confidence)
- PyTorch DDP Tutorial — https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- PyTorch Distributed Overview — https://pytorch.org/docs/stable/distributed.html
- PyTorch Multi-GPU DDP Series — https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html

### Secondary (HIGH confidence)
- Distributed Training with torchrun — https://medium.com/the-owl/the-practical-guide-to-distributed-training-using-pytorch-part-2-on-a-single-node-using-torchrun-9e794baa0410
- PyTorch Distributed Sampler set_epoch — https://discuss.pytorch.org/t/why-is-sampler-set-epoch-epoch-needed-for-distributedsampler/149672

### Tertiary (MEDIUM confidence)
- NCCL Troubleshooting — https://discuss.pytorch.org/t/distributeddataparallel-init-hangs/214091 (community-reported fixes)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Verified from pyproject.toml (PyTorch 2.10.0) and official docs
- Architecture: HIGH - Standard PyTorch DDP patterns from official tutorials
- Pitfalls: HIGH - Multiple community and official sources confirm common issues

**Research date:** 2026-03-16
**Valid until:** 2026-04-16 (30 days for stable PyTorch APIs)
