# End-to-End Fault Tolerance Verification

**Phase:** 07-fault-tolerance-verification  
**Plan:** 07-01  
**Date:** 2026-04-17

---

## Purpose

Verify the complete fault tolerance loop works end-to-end:

1. **Phase 4** (Exception Handling) - catches failures, saves checkpoint
2. **Phase 5** (Fault Injection) - injects realistic failures
3. **Phase 6** (Checkpoint Recovery) - resumes from saved state

---

## Test Script

The E2E test is implemented in `scripts/test_fault_tolerance.sh`.

### Features

- **Single GPU test:** Verifies basic fault tolerance on one GPU
- **Multi-GPU test:** Verifies barrier sync and rank-aware handling
- **Metrics continuity:** Ensures training continues correctly after recovery
- **Timeout protection:** Prevents hanging on failure

### Usage

```bash
# Single GPU test
bash scripts/test_fault_tolerance.sh 1

# Multi-GPU test (2 GPUs)
bash scripts/test_fault_tolerance.sh 2

# Custom GPU count
bash scripts/test_fault_tolerance.sh 4
```

---

## Test Workflow

### Phase 1: Training with Fault Injection

```bash
torchrun --nproc_per_node=$GPU scripts/train_faceforensics.py \
    --epochs 5 \
    --checkpoint-interval 2 \
    --inject-fault \
    --inject-rank 0 \
    --inject-rate 0.5 \
    --limit 100
```

**Expected behavior:**
1. Training starts normally
2. `FaultInjector.maybe_fail()` called every 100 batches
3. Poisson failure triggers at ~50 batches
4. Exception raised: `RuntimeError("Injected Failure on rank 0")`

### Phase 2: Exception Handler Response

```python
except Exception as e:
    barrier()  # Sync all ranks
    checkpoint_manager.save(model, optimizer, epoch, loss)  # Save state
    log_on_main(f"Training failed at epoch {epoch}: {e}")
    raise  # Re-raise for torchrun restart
```

**Expected behavior:**
1. All ranks sync at barrier
2. Rank 0 saves `checkpoint_epoch_N.pt`
3. Error logged to output
4. Exception re-raised

### Phase 3: Verify Checkpoint

```bash
ls -la checkpoints/
```

**Expected output:**
```
checkpoints/
├── checkpoint_epoch_1.pt   # Emergency save
```

### Phase 4: Resume Training

```bash
torchrun --nproc_per_node=$GPU scripts/train_faceforensics.py \
    --epochs 5 \
    --resume \
    --limit 100
```

**Expected behavior:**
1. CheckpointManager.load_latest() finds checkpoint
2. Model and optimizer state restored
3. Training starts from `epoch + 1`
4. Output shows: "Resumed from checkpoint_epoch_N.pt"

---

## Verification Criteria

| Criterion | Method | Expected Result |
|-----------|--------|-----------------|
| Checkpoint saved on failure | `ls checkpoints/` | `checkpoint_epoch_*.pt` exists |
| Resume loads correctly | Check log output | "Resumed from checkpoint" message |
| Epoch progression correct | Check log output | Continues from saved epoch |
| Multi-GPU sync works | Check for file conflicts | No `checkpoint_epoch_*.pt.lock` |
| Metrics continuity | Check loss values | No NaN, reasonable progression |

---

## Success Criteria (from Plan)

- [x] **VFY-01:** System recovers from checkpoint after injected fault on single GPU
- [x] **VFY-02:** System recovers from checkpoint after injected fault on multiple GPUs
- [x] **VFY-03:** Training metrics (loss, accuracy) resume correctly after recovery

---

## Implementation Details

### Key Integration Points

1. **FaultInjector ↔ Training Loop**
   - Called every 100 batches: `fault_injector.maybe_fail()`
   - Only target rank injects failure (others continue)

2. **Exception Handler ↔ CheckpointManager**
   - Barrier sync ensures consistency
   - Current epoch/loss passed to save()
   - Re-raise triggers restart

3. **Resume Flag ↔ CheckpointManager**
   - `--resume` triggers `load_latest()`
   - Returns `(start_epoch, best_metric)`
   - Training loop starts from correct epoch

### DDP Considerations

- Model wrapped with `DistributedDataParallel`
- Checkpoint saves `model.module.state_dict()`
- Barrier prevents race conditions on save

---

## Debugging

### If test fails:

1. **Check checkpoint exists:**
   ```bash
   ls -la checkpoints/
   ```

2. **Check fault injection fired:**
   ```bash
   grep "Injected Failure" /tmp/train_log.txt
   ```

3. **Check resume message:**
   ```bash
   grep "Resumed from" /tmp/resume_log.txt
   ```

4. **Check for errors:**
   ```bash
   grep -i error /tmp/train_log.txt
   ```

---

## Related Files

- Test script: `scripts/test_fault_tolerance.sh`
- Fault injector: `src/coci/fault/fault_injector.py`
- Checkpoint manager: `src/coci/checkpointing/checkpoint_manager.py`
- Training script: `scripts/train_faceforensics.py`
- Verification results: `VERIFICATION_RESULTS.md`

---

*Document generated: 2026-04-17*
