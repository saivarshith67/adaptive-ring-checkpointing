# Technology Stack: Multi-GPU Fault Injection & Recovery

**Project:** Adaptive Ring Checkpointing v1.1 Fault Tolerance
**Researched:** 2026-04-17
**Confidence:** HIGH

## Recommended Stack

### Core Framework
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `torchrun` | Built-in PyTorch 2.0+ | Fault-tolerant launch | Native support, no new deps |
| `torch.distributed` | Built-in | Process group management | Already in use |

### No New Libraries Required (Recommended)

**Approach: torchrun + Snapshot Pattern**
- Zero additional dependencies
- Uses existing `CheckpointManager`
- Recovery via full process restart from snapshot
- **Best for v1.1 milestone validation**

### Infrastructure
| Component | Role | Integration |
|-----------|------|-------------|
| Existing `CheckpointManager` | Snapshot save/load | Already DDP-compatible |
| Existing `FaultInjector` | Fault simulation | Needs rank-awareness |
| torchrun | Process launch & recovery | Replaces `mp.spawn` |

## Fault Tolerance Additions (v1.1)

### Environment Variables for Fault Monitoring

```bash
# Enable async error handling (faster failure detection)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Set heartbeat timeout (default 10 min, reduce for testing)
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=30
```

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Launch method | `torchrun` | `mp.spawn` | torchrun handles fault recovery automatically |
| Fault tolerance | Snapshot-based | torchft per-step | torchft is complex, nightly-only, overkill for milestone |
| Process cleanup | Let torchrun handle | Manual `destroy_process_group()` | Known NCCL race conditions with manual cleanup |

## Optional: torchft (Future Enhancement)

**Only if per-step fault tolerance is needed (no full restart):**
```bash
pip install torchft-nightly  # or pip install -e '.[dev]' from source
```

torchft provides:
- Per-step fault tolerance (continuous recovery)
- Lighthouse server for coordination
- Peer-to-peer checkpoint transfer

**Not recommended for v1.1 due to complexity and nightly-only status.**

## Installation

### No new installations needed for torchrun approach

Existing dependencies already support fault tolerance:
```bash
# Already in environment
torch>=2.0.0  # includes torchrun, torch.distributed
```

## Integration with Existing Code

### Changes Required

1. **Training script launch** — Replace `mp.spawn` with `torchrun` invocation
2. **Snapshot pattern** — Add snapshot save/load to training loop (extends CheckpointManager)
3. **FaultInjector enhancement** — Add rank-awareness for multi-GPU injection

### What NOT to Change

- `CheckpointManager.save()` logic — already works with DDP
- `CheckpointManager.load_latest()` — resumes from checkpoint correctly
- RingCheckpoint strategy — compatible with snapshot pattern

## Launch Command Changes

### Before (mp.spawn)
```bash
python train_distributed.py --world_size 4
```

### After (torchrun)
```bash
torchrun --standalone --nproc_per_node=4 train_distributed.py
```

**Key differences:**
- torchrun auto-sets RANK, WORLD_SIZE, LOCAL_RANK env vars
- torchrun monitors for failures and restarts from snapshot
- `--standalone` enables single-node mode

## Sources

- [PyTorch torchrun documentation](https://pytorch.org/docs/stable/elastic/run.html)
- [PyTorch Fault Tolerance Tutorial](https://pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html)
- [torchft repository](https://github.com/pytorch/torchft)
