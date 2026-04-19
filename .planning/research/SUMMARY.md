# Research Summary: Multi-GPU Fault Injection and Recovery Integration

**Domain:** PyTorch DDP Fault Tolerance
**Researched:** 2026-04-17
**Overall confidence:** HIGH

---

## Executive Summary

Adding multi-GPU fault injection and checkpoint recovery requires **minimal stack additions**. PyTorch's native `torchrun` provides built-in fault tolerance through snapshot-based recovery — no external libraries are strictly required.

The existing `CheckpointManager` already supports DDP-compatible checkpointing via `model.module.state_dict()`. The gap is in:
1. **Training script restructuring** to use `torchrun` for fault handling
2. **Multi-GPU-aware fault injection** (existing `FaultInjector` is single-process only)
3. **Proper exception propagation** to trigger torchrun recovery

**Key integration points identified:**
- Training loop boundary: `try/except` with exception propagation to torchrun
- Snapshot pattern: Model state + optimizer state + epochs_run
- Rank-aware fault injection: Only inject on subset of ranks
- NCCL watchdog: Built into torchrun, handles GPU timeout detection

---

## Key Findings

### Recommended Stack (Extends Existing)

| Technology | Purpose | Why |
|-----------|---------|-----|
| `torchrun` | Fault-tolerant launch | Built into PyTorch 2.0+, replaces `mp.spawn` |
| Existing `CheckpointManager` | Snapshot save/load | Already DDP-compatible |
| Existing `FaultInjector` (modified) | Fault injection | Needs rank-awareness |

### torchft Alternative

Meta's `torchft` library provides production-grade fault tolerance (per-step recovery, peer-to-peer weight transfer) but introduces significant complexity:
- Requires lighthouse server
- Nightly-only builds (unstable API)
- **Recommendation:** Defer to future milestone; torchrun + snapshots is sufficient for v1.1

---

## Implications for Roadmap

### Phase 1: torchrun Integration

**Goal:** Enable fault detection and automatic restart

**Modifications required:**
- Replace `mp.spawn` launch with `torchrun` invocation
- Add snapshot save/load pattern (extends CheckpointManager)
- Structure training loop for graceful restarts

**Addresses:** Basic fault recovery infrastructure

### Phase 2: Multi-GPU Fault Injection

**Goal:** Inject failures that simulate GPU/process failures

**Modifications required:**
- Modify `FaultInjector` with `target_ranks` parameter
- Add rank-aware injection (subset of ranks, not all)
- Integrate into training loop with proper exception propagation

**Avoids:** Injecting on all ranks (causes deadlock)

### Phase 3: Recovery Verification

**Goal:** End-to-end fault tolerance validation

**Components required:**
- Verify `CheckpointManager.load_latest()` works post-restart
- Test with RingCheckpoint strategy
- Validate epoch tracking for resume position

---

## Critical Pitfalls for v1.1 (from PyTorch GitHub Issues)

### Model Divergence After Recovery (Issue #276 - torchft)
- DDP models end up with **different weights** when training interrupted and resumed
- Recovering rank may skip samples while still participating in gradient averaging
- **Prevention:** Epoch-aware recovery, checkpoint dataloader state, barrier validation

### Optimizer State Loss (Issues #3971, #124546)
- Lazy optimizer initialization causes missing state_dict keys on load
- FSDP parameter sharding changes optimizer state structure
- **Prevention:** Prime optimizer before checkpointing, use DefaultLoadPlanner, load model before optimizer

### Async Checkpoint Race Conditions (Issue #159700)
- Concurrent `async_save()` calls cause indefinite hangs
- **Prevention:** Serialize with `handle.result()`, never concurrent saves

### NCCL Error Masking (Issue #122529)
- Checkpoint loading hides actual storage error behind NCCL timeout
- **Prevention:** Custom error propagation in storage readers, shorter timeouts during checkpoint ops

### CPU Tensor Async Corruption (Issue #144657)
- Optimizer updates modify model before async save completes
- **Prevention:** Wait for `handle.result()` before optimizer step, or use GPU tensors

---

## Confidence Assessment

| Area | Confidence | Reason |
|------|------------|--------|
| Stack | HIGH | torchrun is built-in, documented, stable |
| Integration | HIGH | CheckpointManager already DDP-compatible |
| Fault patterns | MEDIUM | torchft patterns are newer (2024), fewer real-world reports |
| NCCL edge cases | MEDIUM | Known issues with process group cleanup, but torchrun handles |
| Model divergence | HIGH | Issue #276 documented, root cause understood |
| Optimizer state | HIGH | Multiple issues confirmed, solutions documented |
| Async races | HIGH | Issue #159700 confirmed, serialization pattern clear |

---

## Gaps to Address

- **torchft maturity:** Still in active development, may have API changes before stable release
- **NCCL timeout handling:** Specific GPU/hardware combinations may need timeout tuning
- **Checkpoint consistency with RingCheckpoint:** Need to verify snapshot pattern works with ring coordination
- **Dataloader state persistence:** DistributedSampler state not automatically saved — must be explicit
- **World size flexibility:** Checkpoints tied to GPU count may not load when scaling

---

## Requirements Checklist for v1.1

### Must Have
- [ ] Epoch-aware recovery (prevent data skipping on resume)
- [ ] Sampler state checkpointing (epoch, batch_index)
- [ ] Optimizer priming before first checkpoint (`optimizer.step()` once)
- [ ] Explicit barrier sync around checkpoint saves
- [ ] Serialized async_save (wait for handle.result())
- [ ] Exception propagation to torchrun (for auto-restart)
- [ ] Rank-aware fault injection (target_rank parameter)

### Avoid
- [ ] In-script process group reinitialization (use torchrun lifecycle)
- [ ] Concurrent async_save calls
- [ ] Loading optimizer before model state_dict
- [ ] Loading checkpoint without optimizer state (causes step reset)

---

## Sources

### Primary Sources (HIGH confidence)
- [PyTorch Fault-Tolerant Tutorial](https://pytorch.org/tutorials/beginner/ddp_series_fault_tolerance) — Official, authoritative
- [torchrun Documentation](https://pytorch.org/docs/stable/elastic/run.html) — Official PyTorch docs

### torchft Reference (MEDIUM - for future enhancement)
- [Meta torchft GitHub](https://github.com/pytorch/torchft) — Per-step fault tolerance
- [PyTorch Distributed Issues](https://github.com/pytorch/pytorch/issues?q=is%3Aissue+NCCL+destroy_process_group) — NCCL cleanup patterns

---

*Research completed: 2026-04-17*
*Ready for requirement definition: yes*
