# Requirements: Adaptive Ring Checkpointing

**Defined:** 2026-03-16
**Updated:** 2026-04-18 (v1.2 milestone)
**Core Value:** Enable reliable long-running deep learning training with minimal checkpoint overhead through adaptive ring-based checkpoint strategies.

---

## v1 Requirements (v1.0 - COMPLETED)

### Multi-GPU Training Infrastructure

- [x] **MGPU-01**: Initialize distributed process group for multi-GPU coordination
- [x] **MGPU-02**: Add torchrun-based entry point for proper DDP process launch
- [x] **MGPU-03**: Implement rank-aware code paths (rank 0 vs other ranks)
- [x] **MGPU-04**: Wrap model with DistributedDataParallel (DDP)
- [x] **MGPU-05**: Convert BatchNorm layers to SyncBatchNorm for cross-GPU synchronization

### Data Parallelism

- [x] **DATA-01**: Integrate DistributedSampler for proper data partitioning across GPUs
- [x] **DATA-02**: Synchronize epoch count across all GPU workers
- [x] **DATA-03**: Seed data loading workers identically across ranks for reproducibility

### Checkpoint Integration

- [x] **CKPT-01**: Save checkpoint from model.module.state_dict() (not wrapped DDP model)
- [x] **CKPT-02**: Load checkpoint with map_location to handle multi-GPU to single-GPU migration
- [x] **CKPT-03**: Coordinate checkpoint save operations to run only on rank 0
- [x] **CKPT-04**: Verify checkpoint compatibility with DDP state_dict keys

### Dataset Integration

- [x] **DATA-04**: Integrate kagglehub for downloading DFD dataset
- [x] **DATA-05**: Create dataset loader for Deep Fake Detection image dataset
- [x] **DATA-06**: Add face detection preprocessing using MTCNN from facenet-pytorch
- [x] **DATA-07**: Configure image transforms compatible with face detection output

### Metrics & Logging

- [x] **METR-01**: Implement all-reduce operation for aggregating loss across GPUs
- [x] **METR-02**: Aggregate accuracy metrics across all ranks
- [x] **METR-03**: Ensure logging only occurs on rank 0 to avoid duplicate output

---

## v1.1 Requirements (COMPLETED)

### Fault Injection

- [x] **FLTI-01**: User can specify target rank(s) for fault injection via configuration
- [x] **FLTI-02**: Fault injection raises exception only on specified ranks, not all
- [x] **FLTI-03**: Non-injected ranks continue execution or handle exception gracefully

### Exception Handling

- [x] **EXCP-01**: Training loop wraps epoch execution in try/except block
- [x] **EXCP-02**: Exception triggers dist.barrier() synchronization before checkpoint save
- [x] **EXCP-03**: Rank 0 saves emergency checkpoint during exception handling
- [x] **EXCP-04**: Exception propagates to trigger torchrun auto-restart

### Checkpoint Recovery

- [x] **RCVR-01**: User can resume training from latest checkpoint after failure
- [x] **RCVR-02**: Epoch counter correctly resumes from saved state
- [x] **RCVR-03**: Optimizer state correctly loads from checkpoint
- [x] **RCVR-04**: DistributedSampler state resumes with correct epoch offset

### Verification

- [x] **VFY-01**: System recovers from checkpoint after injected fault on single GPU
- [x] **VFY-02**: System recovers from checkpoint after injected fault on multiple GPUs
- [x] **VFY-03**: Training metrics (loss, accuracy) resume correctly after recovery

---

## v1.2 Requirements (Current Milestone)

### Hash Ring

- [ ] **HR-01**: Hash ring data structure maps shard positions to GPU nodes using consistent hashing
- [ ] **HR-02**: Virtual nodes (100 per physical GPU) for load balancing
- [ ] **HR-03**: SHA256-based position hash function normalized to [0.0, 1.0) range
- [ ] **HR-04**: Shard owner lookup using ring.upper_bound (nearest clockwise node)

### Shard Manager

- [x] **SHrd-01**: Shard lifecycle state machine (UNCACHED → CACHING → CACHED → ORPHANED → RE-CACHED)
- [x] **SHrd-02**: Shard metadata tracking (owner, status, paths, last_verified)
- [x] **SHrd-03**: Shard assignment to nodes via hash ring
- [x] **SHrd-04**: Local NVMe cache management for shards

### Fault Detector

- [x] **FLTD-01**: Timeout-based failure detection with configurable TTL and timeout_count
- [x] **FLTD-02**: Node status tracking (ALIVE | SUSPECTED | DEAD)
- [x] **FLTD-03**: Gossip-based failure broadcast to peers
- [x] **FLTD-04**: Quorum-based failure confirmation (51% required)

### Elastic Recaching

- [ ] **ELRC-01**: Orphaned shard detection after node failure
- [ ] **ELRC-02**: One-time central storage fetch per lost shard (not every epoch)
- [ ] **ELRC-03**: Recaching with local NVMe serve after first fetch
- [ ] **ELRC-04**: Shard re-assignment to new owners via hash ring

### Recovery Scheduling

- [ ] **RCV-01**: Epoch rollback protocol to last clean epoch
- [ ] **RCV-02**: Resume mode selection (HOT_RESUME / COLD_RESUME / CRITICAL_FAILURE)
- [ ] **RCV-03**: Work slice rebalancing across surviving nodes
- [ ] **RCV-04**: Coordinated checkpoint protocol with manifest

---

## v2 Requirements

### Advanced Features

- **MGPU-06**: Add mixed precision (AMP) training support
- **CKPT-05**: Implement asynchronous checkpoint saving
- **CKPT-06**: Add checkpoint resharding for changing GPU counts
- **DATA-08**: Add data augmentation using albumentations
- **FLTI-04**: Add real-time GPU health monitoring (memory, temperature)
- **FLTI-05**: Add automatic failover when NCCL timeout detected

---

## v1.2 Traceability (Phases 8-12)

| Requirement | Phase | Status |
|-------------|-------|--------|
| HR-01 | Phase 8 | Pending |
| HR-02 | Phase 8 | Pending |
| HR-03 | Phase 8 | Pending |
| HR-04 | Phase 8 | Pending |
| SHrd-01 | Phase 9 | Complete |
| SHrd-02 | Phase 9 | Complete |
| SHrd-03 | Phase 9 | Complete |
| SHrd-04 | Phase 9 | Complete |
| FLTD-01 | Phase 10 | Complete |
| FLTD-02 | Phase 10 | Complete |
| FLTD-03 | Phase 10 | Complete |
| FLTD-04 | Phase 10 | Complete |
| ELRC-01 | Phase 11 | Pending |
| ELRC-02 | Phase 11 | Pending |
| ELRC-03 | Phase 11 | Pending |
| ELRC-04 | Phase 11 | Pending |
| RCV-01 | Phase 12 | Pending |
| RCV-02 | Phase 12 | Pending |
| RCV-03 | Phase 12 | Pending |
| RCV-04 | Phase 12 | Pending |

**Coverage:**
- v1.2 requirements: 20 total
- Mapped to phases: 20 (5 phases)
- Unmapped: 0 ✓

---

## Out of Scope

| Feature | Reason |
|---------|--------|
| Multi-node training | Single machine multi-GPU only for v1 |
| FSDP (Fully Sharded Data Parallel) | DDP sufficient for v1 scope |
| Real-time deep fake inference | Not part of this training pipeline project |
| DataParallel (legacy) | DDP is the recommended approach |
| Per-step recovery (torchft) | Overkill vs checkpoint-based recovery |
| In-script process group reinit | Use torchrun lifecycle instead |

---

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FLTI-01 | Phase 5 | Complete |
| FLTI-02 | Phase 5 | Complete |
| FLTI-03 | Phase 5 | Complete |
| EXCP-01 | Phase 4 | Complete |
| EXCP-02 | Phase 4 | Complete |
| EXCP-03 | Phase 4 | Complete |
| EXCP-04 | Phase 4 | Complete |
| RCVR-01 | Phase 6 | Complete |
| RCVR-02 | Phase 6 | Complete |
| RCVR-03 | Phase 6 | Complete |
| RCVR-04 | Phase 6 | Complete |
| VFY-01 | Phase 7 | Complete |
| VFY-02 | Phase 7 | Complete |
| VFY-03 | Phase 7 | Complete | |

**Coverage:**
- v1.1 requirements: 12 total
- Mapped to phases: 12 ✓
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-16*
*Last updated: 2026-04-17 after v1.1 roadmap created*