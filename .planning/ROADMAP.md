# Roadmap: Adaptive Ring Checkpointing

**Created:** 2026-03-16
**Updated:** 2026-04-18 (v1.2 milestone)
**Core Value:** Enable reliable long-running deep learning training with minimal checkpoint overhead through adaptive ring-based checkpoint strategies.
**Granularity:** Coarse (3-5 phases)

---

## Phases

### v1.0 Milestone (Completed)

- [x] **Phase 1: Distributed Infrastructure** - Initialize multi-GPU training foundation with process group and data distribution
- [x] **Phase 2: DDP Model & Checkpoint Integration** - Wrap model with DDP, implement rank-aware checkpointing and metrics aggregation
- [x] **Phase 3: Dataset Integration** - Integrate DFD dataset with face detection preprocessing

### v1.1 Milestone (CURRENT - COMPLETED)

- [x] **Phase 4: Exception Handling** - Wrap training loop with try/except for fault detection and graceful handling (completed 2026-04-17)
- [x] **Phase 5: Multi-GPU Fault Injection** - Add rank-aware fault injection to simulate node/GPU failures (completed 2026-04-17)
- [x] **Phase 6: Checkpoint Recovery** - Resume training from checkpoint after failure detected (completed 2026-04-17)
- [x] **Phase 7: Fault Tolerance Verification** - End-to-end fault injection and recovery testing (completed 2026-04-17)

### v1.2 Milestone (CURRENT)

- [x] **Phase 8: Hash Ring** - Implement consistent hashing with virtual nodes for shard management
  (completed 2026-04-18)
- [x] **Phase 9: Shard Manager** - Implement checkpoint shard lifecycle and NVMe caching
  (completed 2026-04-18)
- [x] **Phase 10: Fault Detector** - Implement timeout-based failure detection with gossip quorum
  (completed 2026-04-18)
- [ ] **Phase 11: Elastic Recaching** - Implement one-time central storage recaching after failures
- [ ] **Phase 12: Recovery Scheduling** - Implement epoch rollback and work rebalancing

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

### Phase 4: Exception Handling (v1.1 - COMPLETED)

**Goal:** Wrap training loop with try/except for fault detection and graceful handling

**Depends on:** Phase 3 (v1.0 complete)

**Requirements:** EXCP-01, EXCP-02, EXCP-03, EXCP-04

**Success Criteria** (what must be TRUE):
1. Training loop wraps epoch execution in try/except block
2. Exception triggers dist.barrier() synchronization before checkpoint save
3. Rank 0 saves emergency checkpoint during exception handling
4. Exception propagates to trigger torchrun auto-restart
5. Other ranks wait at barrier during exception handling (no hang)

**Plans:** 1/1 plans complete

- [x] 04-01-PLAN.md — Try/except wrapper, barrier sync, emergency checkpoint

---

### Phase 5: Multi-GPU Fault Injection (v1.1 - COMPLETED)

**Goal:** Add rank-aware fault injection to simulate node/GPU failures

**Depends on:** Phase 4

**Requirements:** FLTI-01, FLTI-02, FLTI-03

**Success Criteria** (what must be TRUE):
1. User can specify target rank(s) for fault injection via configuration
2. Fault injection raises exception only on specified ranks
3. Non-injected ranks continue execution or handle exception gracefully
4. Configurable failure timing (every N steps, random, etc.)

**Plans:** 1/1 plans complete

- [x] 05-01-PLAN.md — Rank-aware fault injection, configuration integration

---

### Phase 6: Checkpoint Recovery (v1.1 - IN PROGRESS)

**Goal:** Resume training from checkpoint after failure detected

**Depends on:** Phase 5

**Requirements:** RCVR-01, RCVR-02, RCVR-03, RCVR-04

**Success Criteria** (what must be TRUE):
1. User can resume training from latest checkpoint after failure
2. Epoch counter correctly resumes from saved state
3. Optimizer state correctly loads from checkpoint
4. DistributedSampler state resumes with correct epoch offset
5. Rank-specific state is preserved (model, optimizer, scheduler)

**Plans:** 1/1 plans complete

- [x] 06-01-PLAN.md — Checkpoint loading, epoch recovery, optimizer state

---

### Phase 7: Fault Tolerance Verification (v1.1 - COMPLETED)

**Goal:** End-to-end fault injection and recovery testing

**Depends on:** Phase 6

**Requirements:** VFY-01, VFY-02, VFY-03

**Success Criteria** (what must be TRUE):
1. System recovers from checkpoint after injected fault on single GPU
2. System recovers from checkpoint after injected fault on multiple GPUs
3. Training metrics (loss, accuracy) resume correctly after recovery
4. RingCheckpoint strategy works with fault tolerance

**Plans:** 1/1 plans complete

- [x] 07-01-PLAN.md — E2E fault injection and recovery tests

---

### Phase 8: Hash Ring (v1.2 - IN PROGRESS)

**Goal:** Implement consistent hashing with virtual nodes for shard management

**Depends on:** Phase 7 (v1.1 complete)

**Requirements:** HR-01, HR-02, HR-03, HR-04

**Success Criteria** (what must be TRUE):
1. Hash ring data structure maps shard positions to GPU nodes correctly
2. 100 virtual nodes per physical GPU for load balancing
3. SHA256 hash function normalized to [0.0, 1.0) range
4. Shard owner lookup returns nearest clockwise node

**Plans:** 1/1 plans complete

- [ ] 08-01-PLAN.md — Hash ring with virtual nodes

---

### Phase 9: Shard Manager (v1.2 - COMPLETED)

**Goal:** Implement checkpoint shard lifecycle and NVMe caching

**Depends on:** Phase 8

**Requirements:** SHrd-01, SHrd-02, SHrd-03, SHrd-04

**Success Criteria** (what must be TRUE):
1. Shard lifecycle state machine works (UNCACHED → CACHING → CACHED → ORPHANED → RE-CACHED)
2. Shard metadata tracking includes owner, status, paths, last_verified
3. Shard assignment via hash ring is consistent
4. Local NVMe cache management for shards

**Plans:** 1/1 plans complete

- [x] 09-01-PLAN.md — Shard lifecycle and cache

---

### Phase 10: Fault Detector (v1.2 - COMPLETED)

**Goal:** Implement timeout-based failure detection with gossip quorum

**Depends on:** Phase 9

**Requirements:** FLTD-01, FLTD-02, FLTD-03, FLTD-04

**Success Criteria** (what must be TRUE):
1. Timeout-based detection with configurable TTL
2. Node status (ALIVE | SUSPECTED | DEAD) tracking
3. Gossip-based failure broadcast to peers
4. Quorum-based confirmation (51% required)

**Plans:** 1/1 plans complete

- [x] 10-fault-detector-01-PLAN.md — Fault detection with quorum (completed 2026-04-18)

---

### Phase 11: Elastic Recaching (v1.2 - PLANNED)

**Goal:** Implement one-time central storage recaching after failures

**Depends on:** Phase 10

**Requirements:** ELRC-01, ELRC-02, ELRC-03, ELRC-04

**Success Criteria** (what must be TRUE):
1. Orphaned shards detected after node failure
2. One central storage access per lost shard (key optimization)
3. Local NVMe serve after first fetch
4. Shard re-assignment to new owners via hash ring

**Plans:** 1/1 plans

- [ ] 11-01-PLAN.md — Elastic recaching

---

### Phase 12: Recovery Scheduling (v1.2 - PLANNED)

**Goal:** Implement epoch rollback and work rebalancing

**Depends on:** Phase 11

**Requirements:** RCV-01, RCV-02, RCV-03, RCV-04

**Success Criteria** (what must be TRUE):
1. Epoch rollback to last clean epoch
2. Resume mode selection (HOT/COLD/CRITICAL)
3. Work slice rebalancing across surviving nodes
4. Coordinated checkpoint protocol with manifest

**Plans:** 1/1 plans

- [ ] 12-01-PLAN.md — Recovery scheduling

---

## Progress Table

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Distributed Infrastructure | 2/2 | Completed | 2026-03-16 |
| 2. DDP Model & Checkpoint | 1/1 | Completed | 2026-03-16 |
| 3. Dataset Integration | 1/1 | Completed | 2026-03-16 |
| 4. Exception Handling | 1/1 | Completed | 2026-04-17 |
| 5. Multi-GPU Fault Injection | 1/1 | Completed | 2026-04-17 |
| 6. Checkpoint Recovery | 1/1 | Completed | 2026-04-17 |
| 7. Fault Tolerance Verification | 1/1 | Completed | 2026-04-17 |
| 8. Hash Ring | 1/1 | Complete    | 2026-04-18 |
| 9. Shard Manager | 1/1 | Complete    | 2026-04-18 |
| 10. Fault Detector | 1/1 | Complete   | 2026-04-18 |
| 11. Elastic Recaching | 0/1 | Pending | — |
| 12. Recovery Scheduling | 0/1 | Pending | — |

---

## Coverage

### v1.0 (Completed)

**Total v1.0 Requirements:** 19
**Mapped to Phases:** 19 ✓
**Orphaned:** 0 ✓

### v1.1 (Completed)

**Total v1.1 Requirements:** 12
**Mapped to Phases:** 12 ✓
**Orphaned:** 0 ✓

| Phase | Requirements | Count |
|-------|--------------|-------|
| 4 - Exception Handling | EXCP-01, EXCP-02, EXCP-03, EXCP-04 | 4 |
| 5 - Multi-GPU Fault Injection | FLTI-01, FLTI-02, FLTI-03 | 3 |
| 6 - Checkpoint Recovery | RCVR-01, RCVR-02, RCVR-03, RCVR-04 | 4 |
| 7 - Fault Tolerance Verification | VFY-01, VFY-02, VFY-03 | 3 |

### v1.2 (Current Milestone)

**Total v1.2 Requirements:** 20
**Mapped to Phases:** 20 (5 phases)
**Orphaned:** 0 ✓

| Phase | Requirements | Count |
|-------|--------------|-------|
| 8 - Hash Ring | HR-01, HR-02, HR-03, HR-04 | 4 |
| 9 - Shard Manager | SHrd-01, SHrd-02, SHrd-03, SHrd-04 | 4 |
| 10 - Fault Detector | FLTD-01, FLTD-02, FLTD-03, FLTD-04 | 4 |
| 11 - Elastic Recaching | ELRC-01, ELRC-02, ELRC-03, ELRC-04 | 4 |
| 12 - Recovery Scheduling | RCV-01, RCV-02, RCV-03, RCV-04 | 4 |

---

## Notes

- **Phase ordering rationale:** Hash ring → Shard Manager → Fault Detector → Elastic Recaching → Recovery Scheduling
- **Dependency chain:** Consistent hashing (Phase 8) → Shard lifecycle (Phase 9) → Fault detection (Phase 10) → Recaching (Phase 11) → Recovery (Phase 12)
- **HASH_RING.md spec:** Based on SC'24 paper - 100 virtual nodes, one PFS access per lost shard
- **Key optimization:** Each orphaned shard triggers exactly ONE central storage access (vs naive: every epoch)

---
*Roadmap created: 2026-03-16*
*Updated: 2026-04-18 for v1.2 milestone*