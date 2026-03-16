# Roadmap: Adaptive Ring Checkpointing

**Created:** 2026-03-16
**Core Value:** Enable reliable long-running deep learning training with minimal checkpoint overhead through adaptive ring-based checkpoint strategies.
**Granularity:** Coarse (3-5 phases)

---

## Phases

- [x] **Phase 1: Distributed Infrastructure** - Initialize multi-GPU training foundation with process group and data distribution
- [ ] **Phase 2: DDP Model & Checkpoint Integration** - Wrap model with DDP, implement rank-aware checkpointing and metrics aggregation
- [ ] **Phase 3: Dataset Integration** - Integrate DFD dataset with face detection preprocessing

---

## Phase Details

### Phase 1: Distributed Infrastructure

**Goal:** Initialize multi-GPU training foundation with process group and proper data distribution

**Depends on:** Nothing (first phase)

**Requirements:** MGPU-01, MGPU-02, MGPU-03, DATA-01, DATA-02, METR-03

**Success Criteria** (what must be TRUE):
1. Training can launch via torchrun with multiple GPU processes (verified by seeing N processes in NCCL init)
2. Each process correctly identifies its rank and local_rank (GPU device)
3. DistributedSampler partitions CIFAR-100 data so no two GPUs receive the same batch
4. Only rank 0 prints logs to console (no duplicate output)
5. Epoch counter is synchronized across all GPU workers

**Plans:** 2 plans

- [x] 01-distributed-infrastructure-01-PLAN.md — Distributed infrastructure module and torchrun entry point
- [x] 01-distributed-infrastructure-02-PLAN.md — DistributedSampler integration and rank-aware logging

---

### Phase 2: DDP Model & Checkpoint Integration

**Goal:** Wrap model with DDP, implement rank-aware checkpointing and cross-GPU metrics aggregation

**Depends on:** Phase 1

**Requirements:** MGPU-04, MGPU-05, DATA-03, CKPT-01, CKPT-02, CKPT-03, METR-01, METR-02

**Success Criteria** (what must be TRUE):
1. Model is successfully wrapped with DistributedDataParallel and trains with gradient sync
2. BatchNorm layers are converted to SyncBatchNorm (verified via state inspection)
3. Loss values are correctly aggregated across GPUs (all-reduce produces same value on all ranks)
4. Accuracy metrics are correctly aggregated across all ranks
5. Checkpoint saves successfully from rank 0 only (other ranks skip save)
6. Saved checkpoint can be loaded back and training resumes correctly
7. Checkpoint state_dict is compatible with DDP (no "module." prefix issues)

**Plans:** TBD

---

### Phase 3: Dataset Integration

**Goal:** Integrate DFD dataset with face detection preprocessing and optimize data pipeline

**Depends on:** Phase 2

**Requirements:** CKPT-04, DATA-04, DATA-05, DATA-06, DATA-07

**Success Criteria** (what must be TRUE):
1. DFD dataset downloads successfully via kagglehub
2. Dataset loader creates proper image batches from DFD face images
3. MTCNN face detection runs on images and produces face bounding boxes
4. Image transforms produce correctly sized tensors for model input
5. Checkpoint can save/load successfully when using DFD dataset

**Plans:** TBD

---

## Progress Table

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Distributed Infrastructure | 2/2 | Completed | 2026-03-16 |
| 2. DDP Model & Checkpoint Integration | 0/1 | Not started | - |
| 3. Dataset Integration | 0/1 | Not started | - |

---

## Coverage

**Total v1 Requirements:** 19
**Mapped to Phases:** 19 ✓
**Orphaned:** 0 ✓

| Phase | Requirements | Count |
|-------|--------------|-------|
| 1 - Distributed Infrastructure | MGPU-01, MGPU-02, MGPU-03, DATA-01, DATA-02, METR-03 | 6 |
| 2 - DDP Model & Checkpoint Integration | MGPU-04, MGPU-05, DATA-03, CKPT-01, CKPT-02, CKPT-03, METR-01, METR-02 | 8 |
| 3 - Dataset Integration | CKPT-04, DATA-04, DATA-05, DATA-06, DATA-07 | 5 |

---

## Notes

- **Coarse granularity applied:** Combined multi-GPU infrastructure into 3 phases instead of 7
- **Dependency chain:** Process group → DistributedSampler → DDP → Checkpoint integration → Dataset
- **Existing checkpoint strategies:** FixedInterval, AdaptiveInterval, RingCheckpoint work unchanged (only save/load logic needs DDP awareness)

---

*Roadmap created: 2026-03-16*
