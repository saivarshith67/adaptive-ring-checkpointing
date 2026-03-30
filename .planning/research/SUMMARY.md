# Project Research Summary

**Project:** Adaptive Ring Checkpointing - Multi-GPU & Deep Fake Detection
**Domain:** Fault-tolerant Deep Learning Training Systems
**Researched:** 2026-03-16
**Confidence:** HIGH

---

## Executive Summary

This project extends an existing checkpointing system with multi-GPU training support (via PyTorch DistributedDataParallel) and integrates the Deep Fake Detection (DFD) dataset from Kaggle. Research confirms that **DDP is the correct choice** over DataParallel—it uses multi-process architecture avoiding Python's GIL bottleneck and scales properly across GPUs.

The recommended approach layers multi-GPU support on top of the existing checkpoint strategies with three critical adaptations: (1) wrap models in DDP but save from `model.module.state_dict()` to avoid "module." prefix corruption, (2) use DistributedSampler to partition data across GPUs, and (3) coordinate checkpoint decisions across ranks for ring checkpoint integrity. The primary risk is checkpoint/restore failures from incorrect state dict handling—mitigable by testing save/load cycles early in development.

---

## Key Findings

### Recommended Stack

**Core technologies:**
- **PyTorch 2.7.0+** — Deep learning framework (CUDA 12.6/12.8 support, Python 3.10+)
- **DDP (DistributedDataParallel)** — Multi-GPU training (NOT DataParallel—this is critical)
- **facenet-pytorch 2.5.3+** — MTCNN face detection, GPU-accelerated end-to-end
- **albumentations 1.4+** — Image augmentation, 2-3x faster than torchvision transforms
- **kagglehub** — Official Kaggle API for DFD dataset download
- **NCCL backend** — Required for GPU communication (not Gloo)

**Key version constraints:** Python 3.10+, CUDA 12.6 or 12.8 (12.1 deprecated), torchvision must match PyTorch version.

### Expected Features

**Must have (table stakes):**
- DDP Model Wrapping — Required for gradient synchronization across GPUs
- Process Group Initialization — NCCL backend for GPU coordination
- DistributedSampler — Partition data without overlap across GPUs
- Rank-Aware Checkpointing — Only rank 0 saves; load on all ranks
- Device Placement by Rank — Each process uses its own GPU
- Metrics Aggregation — All-reduce for loss/accuracy across ranks

**Should have (competitive):**
- Face-Specific Preprocessing — Adapt transforms for DFD face images
- DataLoader Optimization — pin_memory, num_workers, persistent_workers
- Gradient Scaling (AMP) — FP16 training for memory/throughput
- Async Checkpoint Saving — Background saves to avoid blocking training

**Defer (v2+):**
- Multi-Node Training — Out of scope, adds significant complexity
- FSDP (Fully Sharded Data Parallel) — Overkill for typical ResNet sizes
- Real-Time Inference Pipeline — Separate project

### Architecture Approach

The architecture follows PyTorch's standard DDP pattern: multi-process (1 per GPU), NCCL for gradient synchronization, DistributedSampler for data partitioning. The existing checkpoint strategies (`FixedInterval`, `AdaptiveInterval`, `RingCheckpoint`) work unchanged—only save/load logic needs DDP awareness.

**Major components:**
1. **Process Group** — Initialize distributed communication (NCCL backend)
2. **DistributedSampler** — Partition data across GPUs, ensure no overlap
3. **DDP Wrapper** — Replicate model, synchronize gradients via all-reduce
4. **Checkpoint Manager** — Save/load with DDP awareness (model.module access)
5. **Metric Aggregator** — Collect loss/accuracy across ranks via all-reduce

### Critical Pitfalls

1. **Checkpoint State Dict Key Mismatch** — Saving `model.state_dict()` creates "module." prefix; use `model.module.state_dict()` instead
2. **Checkpointing Only on Rank 0 Without Coordination** — Use barrier or distributed checkpoint API to prevent desynchronization
3. **NCCL Timeout During Gradient Sync** — Handle uneven batches (drop_last), increase timeout for debugging
4. **BatchNorm Statistics Not Synchronized** — Convert to SyncBatchNorm before DDP wrapping
5. **DataLoader Not Using DistributedSampler** — All GPUs load identical data, no parallelism achieved

---

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Infrastructure Setup
**Rationale:** Foundation required before any multi-GPU work. No dependencies on existing code.
**Delivers:** Distributed entry point (torchrun), process group initialization, rank/local_rank handling
**Addresses:** Process Group Initialization (table stake)
**Avoids:** All subsequent pitfalls require this foundation

### Phase 2: Data Layer
**Rationale:** Must partition data correctly before model training can begin
**Delivers:** DistributedSampler integration, epoch synchronization
**Addresses:** DistributedSampler, DataLoader Worker Seeding
**Avoids:** Pitfall #5 (duplicate data across GPUs)

### Phase 3: Model Layer
**Rationale:** Model must be DDP-wrapped before checkpoint integration
**Delivers:** DDP-wrapped model, SyncBatchNorm conversion
**Addresses:** DDP Model Wrapping
**Avoids:** Pitfall #4 (BatchNorm desync)

### Phase 4: Checkpoint Integration
**Rationale:** Core project value—adaptive ring checkpointing must work with multi-GPU
**Delivers:** Rank-aware checkpoint save/load, proper state dict handling
**Addresses:** Rank-Aware Checkpointing, Metrics Aggregation
**Avoids:** Pitfall #1 (state dict mismatch), Pitfall #2 (rank coordination)

### Phase 5: Metrics & Logging
**Rationale:** Proper loss/accuracy reporting required for training visibility
**Delivers:** All-reduce for metrics, rank 0 file I/O coordination
**Addresses:** Metrics Aggregation

### Phase 6: Dataset Integration
**Rationale:** After multi-GPU training works, integrate DFD face dataset
**Delivers:** Face-specific preprocessing, optimized DataLoader
**Addresses:** Face-Specific Preprocessing, DataLoader Optimization

### Phase 7: Advanced Features (Optional)
**Rationale:** Nice-to-have optimizations after core functionality
**Delivers:** Mixed precision (AMP), async checkpointing, fault-tolerant torchrun
**Addresses:** Gradient Scaling, Async Checkpoint Saving

### Phase Ordering Rationale

- **Dependency chain is strict:** Process group → DistributedSampler → DDP wrapping → Checkpoint integration
- **Testing early:** Pitfalls #1 and #5 are detectable immediately; test checkpoint save/load in Phase 1 or 2
- **Ring coordination:** Phase 4 must implement synchronized checkpoint decisions across ranks (Pitfall #9)

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 4 (Checkpoint Integration):** Complex coordination, distributed checkpoint API (DCP) vs simple barrier approach
- **Phase 7 (Advanced Features):** torchrun elastic discovery, DCP resharding

Phases with standard patterns (skip research-phase):
- **Phase 1-3:** Well-documented DDP patterns, official PyTorch tutorials cover extensively
- **Phase 5:** Standard all-reduce patterns

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Verified on pytorch.org (Feb 2026), DDP is production-standard |
| Features | HIGH | Well-documented DDP requirements, clear table stakes vs differentiators |
| Architecture | HIGH | Standard PyTorch DDP patterns, official documentation |
| Pitfalls | HIGH | Multiple sources confirm critical pitfalls, clear prevention strategies |

**Overall confidence:** HIGH

### Gaps to Address

- **Distributed Checkpoint (DCP) vs Simple Checkpoint:** Phase 4 could use either `torch.save` with barrier or `torch.distributed.checkpoint`. DCP supports resharding but adds complexity. Recommend starting simple, upgrading if needed.
- **DFD Dataset Integration:** Research assumes standard face preprocessing but actual DFD dataset structure (folders, annotations) needs verification before Phase 6.
- **Learning Rate Scaling Strategy:** Linear scaling rule documented but may need warmup tuning. Flag for Phase 7 or during hyperparameter tuning.

---

## Sources

### Primary (HIGH confidence)
- PyTorch DDP Documentation — https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
- PyTorch DDP Tutorial — https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- PyTorch Distributed Checkpoint — https://pytorch.org/docs/stable/distributed.checkpoint.html

### Secondary (HIGH confidence)
- PyTorch DDP Series — https://pytorch.org/tutorials/beginner/ddp_series_multigpu
- facenet-pytorch GitHub — Face detection with MTCNN

### Tertiary (MEDIUM confidence)
- Kaggle DFD Dataset — Dataset structure needs verification during Phase 6
- Community blog posts on DDP best practices — Multiple sources confirm pitfalls

---

*Research completed: 2026-03-16*
*Ready for roadmap: yes*
