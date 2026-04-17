# Roadmap: Adaptive Ring Checkpointing

**Created:** 2026-03-16
**Updated:** 2026-04-17 (v1.1 milestone)
**Core Value:** Enable reliable long-running deep learning training with minimal checkpoint overhead through adaptive ring-based checkpoint strategies.
**Granularity:** Coarse (3-5 phases)

---

## Phases

### v1.0 Milestone (Completed)

- [x] **Phase 1: Distributed Infrastructure** - Initialize multi-GPU training foundation with process group and data distribution
- [x] **Phase 2: DDP Model & Checkpoint Integration** - Wrap model with DDP, implement rank-aware checkpointing and metrics aggregation
- [x] **Phase 3: Dataset Integration** - Integrate DFD dataset with face detection preprocessing

### v1.1 Milestone (Current)

- [ ] **Phase 4: Exception Handling** - Wrap training loop with try/except for fault detection and graceful handling
- [ ] **Phase 5: Multi-GPU Fault Injection** - Add rank-aware fault injection to simulate node/GPU failures
- [ ] **Phase 6: Checkpoint Recovery** - Resume training from checkpoint after failure detected
- [ ] **Phase 7: Fault Tolerance Verification** - End-to-end fault injection and recovery testing

---

## Phase Details

### Phase 1: Distributed Infrastructure (v1.0 - COMPLETED)

**Goal:** Initialize multi-GPU training foundation with process group and proper data distribution

**Depends on:** Nothing (first phase)

**Requirements:** MGPU-01, MGPU-02, MGPU-03, DATA-01, DATA-02, METR-03

**Success Criteria** (what must be TRUE):
1. Training can launch via torchrun with multiple GPU processes
2. Each process correctly identifies its rank and local_rank (GPU device)
3. DistributedSampler partitions data so no two GPUs receive the same batch
4. Only rank 0 prints logs to console
5. Epoch counter is synchronized across all GPU workers

**Plans:** 2 plans (completed)

- [x] 01-distributed-infrastructure-01-PLAN.md
- [x] 01-distributed-infrastructure-02-PLAN.md

---

### Phase 2: DDP Model & Checkpoint Integration (v1.0 - COMPLETED)

**Goal:** Wrap model with DDP, implement rank-aware checkpointing and cross-GPU metrics aggregation

**Depends on:** Phase 1

**Requirements:** MGPU-04, MGPU-05, DATA-03, CKPT-01, CKPT-02, CKPT-03, METR-01, METR-02

**Success Criteria** (what must be TRUE):
1. Model is successfully wrapped with DistributedDataParallel
2. BatchNorm layers are converted to SyncBatchNorm
3. Loss values are correctly aggregated across GPUs
4. Accuracy metrics are correctly aggregated across all ranks
5. Checkpoint saves from rank 0 only
6. Saved checkpoint can be loaded and training resumes correctly
7. Checkpoint state_dict is compatible with DDP

**Plans:** 1 plan (completed)

- [x] 02-ddp-model-checkpoint-integration-01-PLAN.md

---

### Phase 3: Dataset Integration (v1.0 - COMPLETED)

**Goal:** Integrate DFD dataset with face detection preprocessing and optimize data pipeline

**Depends on:** Phase 2

**Requirements:** CKPT-04, DATA-04, DATA-05, DATA-06, DATA-07

**Success Criteria** (what must be TRUE):
1. DFD dataset downloads successfully via kagglehub
2. Dataset loader creates proper image batches from DFD face images
3. MTCNN face detection runs on images
4. Image transforms produce correctly sized tensors
5. Checkpoint can save/load when using DFD dataset

**Plans:** 1 plan (completed)

- [x] 03-dataset-integration-01-PLAN.md

---

### Phase 4: Exception Handling

**Goal:** Wrap training loop with try/except for fault detection and graceful handling

**Depends on:** Phase 3 (v1.0 complete)

**Requirements:** EXCP-01, EXCP-02, EXCP-03, EXCP-04

**Success Criteria** (what must be TRUE):
1. Training loop wraps epoch execution in try/except block
2. Exception triggers dist.barrier() synchronization before checkpoint save
3. Rank 0 saves emergency checkpoint during exception handling
4. Exception propagates to trigger torchrun auto-restart
5. Other ranks wait at barrier during exception handling (no hang)

**Plans:** 1 plan

- [ ] 04-exception-handling-01-PLAN.md — Try/except wrapper, barrier sync, emergency checkpoint

---

### Phase 5: Multi-GPU Fault Injection

**Goal:** Add rank-aware fault injection to simulate node/GPU failures

**Depends on:** Phase 4

**Requirements:** FLTI-01, FLTI-02, FLTI-03

**Success Criteria** (what must be TRUE):
1. User can specify target rank(s) for fault injection via configuration
2. Fault injection raises exception only on specified ranks
3. Non-injected ranks continue execution or handle exception gracefully
4. Configurable failure timing (every N steps, random, etc.)

**Plans:** 1 plan

- [ ] 05-fault-injection-01-PLAN.md — Rank-aware fault injection, configuration integration

---

### Phase 6: Checkpoint Recovery

**Goal:** Resume training from checkpoint after failure detected

**Depends on:** Phase 5

**Requirements:** RCVR-01, RCVR-02, RCVR-03, RCVR-04

**Success Criteria** (what must be TRUE):
1. User can resume training from latest checkpoint after failure
2. Epoch counter correctly resumes from saved state
3. Optimizer state correctly loads from checkpoint
4. DistributedSampler state resumes with correct epoch offset
5. Rank-specific state is preserved (model, optimizer, scheduler)

**Plans:** 1 plan

- [ ] 06-checkpoint-recovery-01-PLAN.md — Checkpoint loading, epoch recovery, optimizer state

---

### Phase 7: Fault Tolerance Verification

**Goal:** End-to-end fault injection and recovery testing

**Depends on:** Phase 6

**Requirements:** VFY-01, VFY-02, VFY-03

**Success Criteria** (what must be TRUE):
1. System recovers from checkpoint after injected fault on single GPU
2. System recovers from checkpoint after injected fault on multiple GPUs
3. Training metrics (loss, accuracy) resume correctly after recovery
4. RingCheckpoint strategy works with fault tolerance

**Plans:** 1 plan

- [ ] 07-fault-tolerance-verification-01-PLAN.md — E2E fault injection and recovery tests

---

## Progress Table

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Distributed Infrastructure | 2/2 | Completed | 2026-03-16 |
| 2. DDP Model & Checkpoint | 1/1 | Completed | 2026-03-16 |
| 3. Dataset Integration | 1/1 | Completed | 2026-03-16 |
| 4. Exception Handling | 0/1 | Planned | |
| 5. Multi-GPU Fault Injection | 0/1 | Planned | |
| 6. Checkpoint Recovery | 0/1 | Planned | |
| 7. Fault Tolerance Verification | 0/1 | Planned | |

---

## Coverage

### v1.0 (Completed)

**Total v1.0 Requirements:** 19
**Mapped to Phases:** 19 ✓
**Orphaned:** 0 ✓

### v1.1 (Current Milestone)

**Total v1.1 Requirements:** 12
**Mapped to Phases:** 12 ✓
**Orphaned:** 0 ✓

| Phase | Requirements | Count |
|-------|--------------|-------|
| 4 - Exception Handling | EXCP-01, EXCP-02, EXCP-03, EXCP-04 | 4 |
| 5 - Multi-GPU Fault Injection | FLTI-01, FLTI-02, FLTI-03 | 3 |
| 6 - Checkpoint Recovery | RCVR-01, RCVR-02, RCVR-03, RCVR-04 | 4 |
| 7 - Fault Tolerance Verification | VFY-01, VFY-02, VFY-03 | 3 |

---

## Notes

- **Phase ordering rationale:** Exception handling is foundation → fault injection builds on it → checkpoint recovery needs both → verification validates all
- **Dependency chain:** Try/except wrapper (Phase 4) → Rank-aware injection (Phase 5) → Checkpoint loading (Phase 6) → E2E test (Phase 7)
- **Research findings:** torchrun is built-in, CheckpointManager already DDP-compatible, FaultInjector needs rank-awareness

---
*Roadmap created: 2026-03-16*
*Updated: 2026-04-17 for v1.1 milestone*