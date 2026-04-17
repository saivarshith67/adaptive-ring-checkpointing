# Domain Pitfalls: PyTorch Multi-GPU Training

**Domain:** Fault-tolerant deep learning training with multi-GPU support
**Researched:** 2026-03-16
**Project:** Adaptive Ring Checkpointing

---

## Critical Pitfalls

Mistakes that cause silent data corruption, training hangs, or checkpoint/restore failures.

### Pitfall 1: Checkpoint State Dict Key Mismatch

**What goes wrong:**  
Saving `model.state_dict()` from a DDP-wrapped model produces keys prefixed with `"module."`, causing load failures when restoring to single-GPU or non-DDP models.

**Why it happens:**  
DDP wraps the original model, so `model.state_dict()` returns the wrapper's state dict with `module.` prefixes. The checkpoint saved from rank 0 cannot be loaded into a non-DDP model.

**Consequences:**
- Training cannot be resumed after interruption
- Saved checkpoints are unusable for inference
- Silent data corruption if keys loosely match

**Prevention:**
```python
# WRONG - creates checkpoint with "module." prefix
torch.save(model.state_dict(), 'checkpoint.pt')

# CORRECT - unwrap before saving
torch.save(model.module.state_dict(), 'checkpoint.pt')
```

**Detection:**  
`RuntimeError: Error(s) in loading state_dict: Missing key(s)... Unexpected key(s): "module.conv1.weight"...`

**Phase Mapping:**  
- **Phase: Multi-GPU Integration** — This is the phase where checkpoint save/load is implemented
- Prevents broken resume capability after faults

---

### Pitfall 2: Checkpointing Only on Rank 0 Without Coordination

**What goes wrong:**  
Saving checkpoints only on rank 0 while other ranks continue training causes desynchronization. If a fault occurs mid-checkpoint, other ranks have advanced further.

**Why it happens:**  
Naive `if rank == 0: save_checkpoint()` pattern. All ranks must coordinate or use distributed checkpoint APIs.

**Consequences:**
- Inconsistent checkpoint state across ranks
- Training resumes with inconsistent model weights
- Ring checkpoint strategy breaks if ranks are misaligned

**Prevention:**
```python
# Use distributed checkpoint API
import torch.distributed.checkpoint as dcp

# All ranks save coordinated
dcp.save_state_dict(state_dict, ...)

# OR synchronize before rank-0 save
dist.barrier()
if rank == 0:
    save_checkpoint()
```

**Detection:**  
Checkpoints load with mismatched optimizer states or epoch numbers across runs.

**Phase Mapping:**  
- **Phase: Multi-GPU Checkpoint Coordination** — Critical for fault tolerance
- Directly impacts adaptive ring checkpoint reliability

---

### Pitfall 3: NCCL Timeout During Gradient Synchronization

**What goes wrong:**  
Training hangs with NCCL timeout errors during gradient all-reduce. Most common during first few batches or at validation.

**Why it happens:**  
- Different batch sizes across ranks (last batch uneven)
- One GPU falls behind (slow data loading, OOM)
- Validation forward pass without proper synchronization

**Consequences:**
- Complete training hang requiring restart
- Lost progress since last checkpoint
- Silent data loss if watchdog kills process mid-save

**Prevention:**
```python
# Handle uneven batches
drop_last=True  # Drop incomplete final batch

# Increase timeout for debugging
import os
os.environ['NCCL_TIMEOUT'] = '3600'  # 1 hour

# Properly synchronize validation
model.eval()
with torch.no_grad():
    # All ranks must execute validation together
    dist.barrier()
    for batch in val_loader:
        ...
model.train()
```

**Detection:**  
`RuntimeError: NCCL timeout: WorkNCCL(...) ran for X milliseconds before timing out`

**Phase Mapping:**  
- **Phase: Multi-GPU Integration** — Network/communication setup
- **Phase: Fault Tolerance Testing** — Verify handles timeouts gracefully

---

### Pitfall 4: BatchNorm Statistics Not Synchronized

**What goes wrong:**  
Training runs but validation accuracy is poor. BatchNorm layers compute statistics only on local batch, not across all GPUs.

**Why it happens:**  
Default `nn.BatchNorm2d` computes stats per GPU. Without SyncBatchNorm, each GPU has different running stats.

**Consequences:**
- Poor validation accuracy (10-20% worse)
- Model performs differently at inference
- Inconsistent checkpoint quality across ranks

**Prevention:**
```python
# Convert BatchNorm to SyncBatchNorm BEFORE wrapping with DDP
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = DDP(model, ...)
```

**Detection:**  
Validation accuracy significantly lower than single-GPU baseline. Model inference varies across GPUs.

**Phase Mapping:**  
- **Phase: Multi-GPU Integration** — Model architecture modifications

---

### Pitfall 5: DataLoader Not Using DistributedSampler

**What goes wrong:**  
All GPUs load the same data, training effectively uses only 1/N GPUs' worth of data per epoch.

**Why it happens:**  
Default DataLoader iterates over entire dataset on each GPU without DistributedSampler.

**Consequences:**
- No speedup from multiple GPUs
- Training sees duplicate batches (N times effective batch size)
- Model overfits to duplicate data

**Prevention:**
```python
from torch.utils.data.distributed import DistributedSampler

train_sampler = DistributedSampler(
    train_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    seed=42,
    drop_last=True  # Prevent uneven batches
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size_per_gpu,
    sampler=train_sampler,
    num_workers=num_workers,
    pin_memory=True,
)

# CRITICAL: Call set_epoch between epochs for proper shuffling
for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)
    for batch in train_loader:
        ...
```

**Detection:**  
Training is as slow as single GPU. Log shows identical batch IDs across ranks.

**Phase Mapping:**  
- **Phase: Multi-GPU Data Pipeline** — DataLoader modification

---

## Moderate Pitfalls

Mistakes that cause performance degradation or subtle bugs.

### Pitfall 6: Learning Rate Not Scaled with Batch Size

**What goes wrong:**  
Training converges slower or diverges when using multiple GPUs. Effective batch size increases N times but learning rate remains the same.

**Why it happens:**  
With N GPUs, effective batch size = batch_size_per_gpu × N. Without LR scaling, each GPU computes gradients for smaller batches.

**Consequences:**
- Slower convergence
- Possible divergence or instability
- Inconsistent final model quality

**Prevention:**
```python
# Linear scaling rule (valid for typical batch sizes)
base_lr = 0.001
base_batch = 32
effective_batch = batch_size_per_gpu * dist.get_world_size()
scaled_lr = base_lr * (effective_batch / base_batch)

# Alternative: use linear warmup + scaling
optimizer = torch.optim.SGD(model.parameters(), lr=scaled_lr, momentum=0.9)
```

**Detection:**  
Loss curves differ significantly from single-GPU training. Divergence after adding GPUs.

**Phase Mapping:**  
- **Phase: Hyperparameter Tuning** — After basic multi-GPU works

---

### Pitfall 7: Random Seed Not Set Per Worker

**What goes wrong:**  
Data augmentation produces identical transforms across all GPU workers, or same random sequence every epoch.

**Why it happens:**  
Workers share the same random seed by default. numpy.random and random modules aren't seeded per worker.

**Consequences:**
- All GPUs see identical augmented images
- Effective data diversity reduced by factor of N
- Training may overfit to specific augmentations

**Prevention:**
```python
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    random.seed(random.getrandbits(32) + worker_id)

DataLoader(
    dataset,
    worker_init_fn=worker_init_fn,
    ...
)
```

**Detection:**  
Identical loss values across all GPUs for several batches. Augmentation produces same results.

**Phase Mapping:**  
- **Phase: Multi-GPU Data Pipeline** — DataLoader configuration

---

### Pitfall 8: Model.eval() Causes Deadlock in Validation

**What goes wrong:**  
Training hangs during first validation with DDP and BatchNorm layers.

**Why it happens:**  
Calling `model.eval()` on DDP-wrapped model with BatchNorm can cause synchronization issues. All ranks must transition together.

**Consequences:**
- Complete training hang during validation
- Requires force kill and restart

**Prevention:**
```python
# Use proper context managers
model.eval()  # All ranks call this

# Wrap validation in no_grad and synchronize
with torch.no_grad():
    dist.barrier()  # Ensure all ranks reach validation together
    for val_batch in val_loader:
        output = model(val_batch)
    dist.barrier()  # Ensure all ranks finish together

model.train()
```

**Detection:**  
Training hangs exactly at first validation step.

**Phase Mapping:**  
- **Phase: Multi-GPU Integration** — Training loop modifications

---

### Pitfall 9: Inconsistent Checkpoint Frequency Across Ranks

**What goes wrong:**  
With adaptive checkpoint strategies, different ranks decide to checkpoint at different times, breaking the ring coordination.

**Why it happens:**  
Each rank computes its own checkpoint trigger based on local statistics, but ring checkpointing requires coordinated decisions.

**Consequences:**
- Ring structure breaks
- Checkpoint data incomplete or inconsistent
- Recovery may fail

**Prevention:**
```python
# Synchronize checkpoint decision across ranks
def should_checkpoint(rank, epoch, step, stats):
    # Each rank computes local decision
    local_trigger = compute_adaptive_trigger(stats)
    
    # All-reduce to get global decision (all ranks must agree)
    trigger_tensor = torch.tensor([local_trigger], device='cuda')
    dist.all_reduce(trigger_tensor, op=dist.ReduceOp.MIN)
    
    return trigger_tensor.item() == 1
```

**Detection:**  
Checkpoint files have mismatched step numbers or ring structure corrupted.

**Phase Mapping:**  
- **Phase: Adaptive Ring Checkpoint Integration** — Core to the project value

---

### Pitfall 10: Checkpoint Restoration Without DDP Unwrapping

**What goes wrong:**  
Loading a single-GPU checkpoint into a DDP model fails silently or causes dimension mismatches.

**Why it happens:**  
Single-GPU checkpoints have no `module.` prefix. DDP expects `module.` prefix or needs special handling.

**Prevention:**
```python
# Load checkpoint into DDP model correctly
checkpoint = torch.load('checkpoint.pt')

# If checkpoint was saved from non-DDP model
model.module.load_state_dict(checkpoint['model_state_dict'])

# If checkpoint was saved from DDP model
# (already has module. prefix, works directly)
# model.load_state_dict(checkpoint['model_state_dict'])

# Then wrap with DDP
model = DDP(model, device_ids=[local_rank])
```

**Detection:**  
`RuntimeError: Missing key(s)... Unexpected key(s): "module.*"`

**Phase Mapping:**  
- **Phase: Multi-GPU Checkpoint Restore** — Fault recovery implementation

---

## Minor Pitfalls

### Pitfall 11: CUDA Graphs Incompatibility with DDP

**What goes wrong:**  
Using `torch.compile` with DDP causes significant slowdowns or hangs with BatchNorm layers.

**Prevention:**  
Disable `reduce_overhead` when using BatchNorm with DDP:
```python
model = DDP(model)
compiled_model = torch.compile(model, mode='reduce-overhead', ...)
# Or avoid compile with BatchNorm + DDP
```

### Pitfall 12: Gradient Accumulation Not Synchronized

**What goes wrong:**  
Gradient accumulation steps differ across ranks, causing desynchronization.

**Prevention:**  
Ensure all ranks complete same number of accumulation steps before optimizer step:
```python
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        # All ranks must sync here
        dist.all_reduce(gradients)
        optimizer.step()
        optimizer.zero_grad()
```

---

## Phase-Specific Warnings

| Phase | Likely Pitfall | Mitigation |
|-------|---------------|------------|
| Multi-GPU Integration | Pitfall 1, 4, 5, 8 | Test checkpoint save/load early; verify SyncBatchNorm; use DistributedSampler |
| Multi-GPU Data Pipeline | Pitfall 7, 11 | Worker init function; proper seed per epoch |
| Adaptive Ring Checkpoint | Pitfall 9 | Coordinate checkpoint decisions across ranks |
| Fault Tolerance Testing | Pitfall 2, 3 | Test with simulated NCCL timeouts; verify checkpoint consistency |
| Hyperparameter Tuning | Pitfall 6 | Scale learning rate with effective batch size |

---

## Sources

- PyTorch DDP Documentation: https://pytorch.org/tutorials/beginner/ddp_series_multigpu
- PyTorch Distributed Checkpoint: https://pytorch.org/docs/stable/distributed.checkpoint.html
- PyTorch Forums: NCCL timeout troubleshooting
- GitHub Issues: #24397 (SyncBatchNorm eval), #571 (checkpoint saving), #647 (module prefix)
- Medium: "The Complete Guide to Multi-Node Distributed Training" (Feb 2026)
- Stack Overflow: DDP state_dict loading issues

---

---

## NEW: Fault Injection & Recovery Pitfalls (v1.1)

### Critical: Uncoordinated Exception Handling

**What goes wrong:**  
One rank raises RuntimeError from fault injection, other ranks continue training, causing barrier deadlock.

**Why it happens:**  
Exception raised on rank 0 but rank 1 continues. Next barrier() or all_reduce in training loop hangs indefinitely waiting for rank 0.

**Consequences:**
- Training hangs indefinitely
- Force kill required
- Progress lost since last checkpoint

**Prevention:**
```python
# Wrap training loop with exception handling
for epoch in range(start_epoch, cfg.epochs):
    try:
        avg_loss = train_epoch(model, train_loader, optimizer, device, epoch, train_sampler)
        
        # Fault injection after forward pass
        if inject_fault and fault_injector:
            fault_injector.maybe_fail()  # May raise RuntimeError
            
    except RuntimeError as e:
        log_on_main(f"Fault detected: {e}")
        
        # Barrier FIRST - ensure all ranks reach consistent state
        if is_distributed_initialized():
            barrier()
        
        # Emergency checkpoint on rank 0
        checkpoint_manager.save(model, optimizer, epoch, avg_loss)
        
        # Cleanup and exit cleanly
        cleanup_distributed()
        raise  # Re-raise for test harness
```

**Detection:**  
Training hangs with no output after "Epoch N" logging. `dist.barrier()` never completes.

**Phase Mapping:**  
- **Phase 2: Exception Handling** — Core to fault tolerance

---

### Critical: Rank-Specific Fault Injection Not Reaching All Ranks

**What goes wrong:**  
Fault injected only on rank 0, but test requires all ranks to fail identically for proper recovery testing.

**Why it happens:**  
Existing `FaultInjector` injects on whichever process calls it. Without rank coordination, only one rank fails.

**Consequences:**
- Single-rank failure (not multi-GPU fault simulation)
- Recovery behavior not properly tested

**Prevention:**
```python
class FaultInjector:
    def __init__(self, failure_rate_per_second=0.02, target_rank=None):
        self.lambda_rate = failure_rate_per_second
        self.target_rank = target_rank  # None = all ranks, or specific rank
        
    def maybe_fail(self, rank):
        # If target_rank is None, inject on ALL ranks
        # If target_rank is specified, inject only on that rank
        if self.target_rank is not None and rank != self.target_rank:
            return
        
        # Poisson failure logic
        ...

# Usage in training loop
if inject_fault and fault_injector:
    fault_injector.maybe_fail(rank)
```

**Detection:**  
Only rank 0 reports "Injected failure" message, other ranks continue.

**Phase Mapping:**  
- **Phase 1: Multi-GPU Fault Injection** — Modify existing FaultInjector

---

### Critical: Checkpoint Save During Exception Not Synchronized

**What goes wrong:**  
Exception triggers checkpoint save but other ranks may have already exited, causing incomplete checkpoint.

**Why it happens:**  
Exception caught but no barrier before checkpoint. Some ranks may have started exiting before rank 0 saves.

**Prevention:**
```python
except RuntimeError as e:
    # BARRIER FIRST - sync all ranks
    if is_distributed_initialized():
        barrier()  # All ranks must reach here
    
    # Now checkpoint - all ranks synchronized
    checkpoint_manager.save(model, optimizer, epoch, avg_loss)
```

**Detection:**  
Checkpoint file corrupted or missing optimizer state.

**Phase Mapping:**  
- **Phase 2: Exception Handling** — Post-exception coordination

---

### Moderate: Missing Epoch Tracking After Recovery

**What goes wrong:**  
Training resumes from epoch 0 after recovery instead of from saved epoch.

**Why it happens:**  
`CheckpointManager.load_latest()` returns epoch + 1 to continue from next epoch, but training loop starts from config.epochs or doesn't use the returned value.

**Prevention:**
```python
# Use the returned epoch from load_latest
start_epoch = checkpoint_manager.load_latest(model, optimizer, device)
# Returns: next epoch to train (epoch from checkpoint + 1)
# If no checkpoint: returns 0 to start fresh

# Continue training from start_epoch
for epoch in range(start_epoch, cfg.epochs):
    ...

# NOTE: load_latest already returns epoch + 1 for resume
```

**Detection:**  
Model retrained from beginning after manual restart, losing progress.

**Phase Mapping:**  
- **Phase 3: Checkpoint Recovery** — Ensure epoch tracking

---

### Moderate: NCCL Watchdog Hangs Without Async Error Handling

**What goes wrong:**  
NCCL timeout occurs but process hangs instead of raising exception.

**Why it happens:**  
Default NCCL watchdog blocks without `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`.

**Prevention:**
```bash
# Set before training
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=60  # 1 minute for testing
```

**Detection:**  
Training hangs with no error message. Process must be force-killed.

**Phase Mapping:**  
- **Phase 1: Fault Injection Setup** — Environment configuration

---

---

## NEW: Critical Fault Tolerance Pitfalls (from PyTorch GitHub Issues)

### Critical: DDP Model Divergence After Recovery

**What goes wrong:** When training is interrupted and resumed, models across ranks end up with different final weights even when starting from the same checkpoint.

**Why it happens:** 
- DDP gradients synchronized via `all_reduce`. When one rank fails and recovers, it may be at different training step than peers.
- Recovering rank may skip samples while still participating in gradient averaging.
- Issue #276 (torchft): *"final models across ranks are different when training is interrupted"*
- `should_commit` returning false for recovery worker causes sample loss

**Consequences:**
- Model divergence between ranks
- Silent data inconsistency (loss looks normal but model quality degrades)

**Prevention:**
- Always checkpoint dataloader state (epoch, batch_index) alongside model/optimizer
- Implement epoch-level recovery: restart from beginning of epoch if recovery would cause data skip
- Use barrier-based epoch validation before continuing training

**Detection:**
```python
# Compare model weights across ranks after recovery:
if dist.get_rank() == 0:
    dist.send(model.module.weight.clone(), dst=1)
else:
    remote_weight = torch.empty_like(model.module.weight)
    dist.recv(remote_weight, src=0)
    if not torch.allclose(model.module.weight, remote_weight):
        raise RuntimeError("Model divergence detected after recovery!")
```

**Phase Mapping:** Phase 3: Checkpoint Recovery — requires epoch-aware recovery design

---

### Critical: Process Group Reinitialization Deadlock

**What goes wrong:** After destroying a process group (e.g., after failure), attempting to reinitialize causes hangs or NCCL errors.

**Why it happens:** 
- `destroy_process_group()` + re-init is *"currently unsupported/untested"* per PyTorch docs
- Synchronization after destroy must use non-torch.distributed primitives
- NCCL backend state persists even after process group destruction

**Prevention:**
- **Avoid reinitializing process groups mid-training**
- Design recovery to use external restart (torchrun with `--max-restarts`) rather than in-script recovery
- Use `torch.distributed.elastic` for fault tolerance rather than manual recovery

**Phase Mapping:** Phase 1: Multi-GPU Fault Injection — prefer torchrun lifecycle management

---

### Critical: Optimizer State Loss During Recovery

**What goes wrong:** After loading checkpoint, optimizer step count or momentum states are incorrect, causing training instability.

**Why it happens:**
- Lazy optimizer initialization: states created on first `step()`, not on model creation
- Loading optimizer state_dict before any optimizer steps fails silently
- Issue #3971: *"Missing key in checkpoint state_dict: optimizer.state.0.step"*
- Issue #124546: FQN mismatch with activation checkpointing

**Prevention:**
```python
# 1. Always prime optimizer before checkpointing
optimizer.step()  # At least once
save_checkpoint({'optimizer': optimizer.state_dict(), ...})

# 2. Load in correct order:
model.load_state_dict(checkpoint['model'])  # Model first
optimizer.load_state_dict(checkpoint['optimizer'])  # Then optimizer

# 3. For FSDP, use explicit planners:
from torch.distributed.checkpoint import DefaultLoadPlanner
planner = DefaultLoadPlanner()  # Don't omit this!
dcp.load(state_dict, planner=planner)
```

**Phase Mapping:** Phase 3: Checkpoint Recovery — verify optimizer state integrity

---

### Critical: Async Checkpoint Race Conditions

**What goes wrong:** Multiple concurrent async checkpoint saves cause hangs or data corruption.

**Why it happens:**
- `torch.distributed.checkpoint.async_save()` reuses a dedicated Gloo process group
- Issue #159700: *"hangs when 2 handles are created"*

**Prevention:**
```python
# WRONG - causes race condition:
handle1 = async_save({'model': model.state_dict()})
handle2 = async_save({'optimizer': optimizer.state_dict()})

# CORRECT - serialize with result():
handle = async_save({
    'model': model.state_dict(), 
    'optimizer': optimizer.state_dict()
})
handle.result()  # Wait before next save or optimizer step
```

**Phase Mapping:** Phase 2: Exception Handling — must serialize async operations

---

### Critical: NCCL Error Masking Root Cause

**What goes wrong:** Checkpoint loading fails with NCCL timeout, hiding the actual storage/network error.

**Why it happens:**
- PyTorch checkpointing uses `all_gather_object()` internally
- Issue #122529: NCCL timeout obscures real exception

**Prevention:**
```python
# Add explicit error handling in custom storage reader:
class DebugStorageWriter(FileSystemWriter):
    def write_data(self, plan):
        try:
            return super().write_data(plan)
        except Exception as e:
            print(f"Rank {dist.get_rank()} write failed: {e}")
            dist.all_reduce(torch.zeros(1))  # Force sync to propagate
            raise
```

**Phase Mapping:** Phase 2: Exception Handling — critical for debugging production failures

---

### Critical: CPU Tensor Async Checkpoint Corruption

**What goes wrong:** Checkpoint saved with CPU tensors contains wrong values after optimizer update.

**Why it happens:**
- Async save uses staging buffer
- Issue #144657: new value written instead of original

**Prevention:**
```python
# Move to GPU before async save:
model_gpu = model.cuda()
handle = dcp.async_save({'model': model_gpu.state_dict()}, ...)
model.cpu()
handle.result()  # Wait before optimizer step
```

**Phase Mapping:** Phase 2: Checkpoint Coordination — timing matters for async saves

---

## Phase-Specific Warnings (Updated)

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|----------------|------------|
| Multi-GPU fault injection | Process group reinit deadlock | Use torchrun lifecycle, not in-script recovery |
| Checkpoint save coordination | Async race conditions | Serialize async_save with handle.result() |
| Recovery state loading | Optimizer state mismatches | Prime optimizer, use DefaultLoadPlanner |
| Dataloader recovery | Sample skipping | Checkpoint sampler epoch/index state |
| torchrun integration | Worker failure handling issues | Test on target PyTorch version |
| Graceful degradation | NCCL error masking | Add explicit error propagation |

---

## Quick Reference for Requirement Definition

**Must have for fault tolerance:**
- [ ] Epoch-aware recovery (prevent data skipping)
- [ ] Sampler state checkpointing
- [ ] Optimizer priming before first checkpoint
- [ ] Explicit barrier synchronization around saves
- [ ] Error propagation in custom storage readers
- [ ] Serialized async_save (no concurrent calls)

**Avoid or deprioritize:**
- [ ] In-script process group reinitialization (use torchrun instead)
- [ ] Concurrent async_save calls (serialize with handle.result())
- [ ] Cross-version torchrun fault tolerance testing (test on target version)

---

## Confidence Assessment (Updated)

| Pitfall | Confidence | Reason |
|---------|------------|--------|
| State dict key mismatch | HIGH | Well-documented in PyTorch forums, official docs |
| Rank 0 checkpointing | HIGH | Common issue with clear solutions |
| NCCL timeout | MEDIUM | Multiple causes, environment-dependent |
| BatchNorm sync | HIGH | Official PyTorch recommendation |
| DistributedSampler | HIGH | Required for proper data distribution |
| LR scaling | MEDIUM | Depends on optimizer, warmup strategy |
| Worker seeding | HIGH | Documented in PyTorch |
| Model.eval() deadlock | MEDIUM | Known issue with workarounds |
| Ring coordination | HIGH | Core to ring checkpoint design |
| Checkpoint loading | HIGH | Well-documented patterns |
| Uncoordinated exception | HIGH | Barrier synchronization required |
| Rank-specific injection | MEDIUM | Requires modification to existing code |
| Checkpoint sync | HIGH | Barrier before save after exception |
| DDP model divergence | HIGH | Issue #276 documented in torchft |
| Process group reinit | HIGH | PyTorch docs explicitly unsupported |
| Optimizer state loss | HIGH | Issues #3971, #124546 |
| Async checkpoint races | HIGH | Issue #159700 |
| NCCL error masking | HIGH | Issue #122529 |
| CPU tensor corruption | HIGH | Issue #144657 |

---

## Sources

- PyTorch GitHub #276 (DDP model divergence)
- PyTorch GitHub #146371, #150916, #147064 (torchrun failures)
- PyTorch GitHub #159700 (async save hang)
- PyTorch GitHub #122529 (NCCL masking)
- PyTorch GitHub #3971, #124546 (optimizer state)
- PyTorch GitHub #144657 (CPU tensor corruption)
- PyTorch GitHub #811 (world size mismatch)
- PyTorch Official Docs - Distributed Checkpointing
- PyTorch Tutorials - Fault-tolerant Distributed Training
- Meta torchft GitHub
- Medium - Multi-Node DDP Troubleshooting (Feb 2026)
