---
phase: 01-distributed-infrastructure
plan: 02
type: execute
wave: 2
depends_on:
  - 01-distributed-infrastructure-01
files_modified:
  - scripts/train_distributed.py
autonomous: true
requirements:
  - DATA-01
  - DATA-02
  - METR-03

must_haves:
  truths:
    - "DistributedSampler partitions CIFAR-100 data so no two GPUs receive the same batch"
    - "Epoch counter is synchronized across all GPU workers via set_epoch()"
    - "Only rank 0 prints logs to console (no duplicate output)"
  artifacts:
    - path: "scripts/train_distributed.py"
      provides: "Distributed training with data partitioning and rank-aware logging"
      contains: ["DistributedSampler", "sampler.set_epoch", "log_on_main"]
  key_links:
    - from: "scripts/train_distributed.py"
      to: "src/coci/distributed.py"
      via: "import"
      pattern: "from src.coci.distributed import.*log_on_main"
    - from: "scripts/train_distributed.py"
      to: "torch.utils.data.distributed.DistributedSampler"
      via: "import and instantiation"
      pattern: "DistributedSampler"
---

<objective>
Integrate DistributedSampler for data partitioning and rank-0-only logging into training script.

Purpose: Enable proper data distribution across GPUs and eliminate duplicate log output.
Output: Updated train_distributed.py with DistributedSampler integration and rank-aware logging
</objective>

<context>
@src/coci/distributed.py (created in Plan 01)
@src/coci/data_ingestor/cifar.py (existing CIFAR dataset)
@scripts/train.py (reference for existing training logic)

# Phase 1 Requirements (from ROADMAP.md)
- DATA-01: Integrate DistributedSampler for proper data partitioning across GPUs
- DATA-02: Synchronize epoch count across all GPU workers
- METR-03: Ensure logging only occurs on rank 0 to avoid duplicate output
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add DistributedSampler integration to training</name>
  <files>scripts/train_distributed.py</files>
  <action>
Update scripts/train_distributed.py to integrate DistributedSampler:

1. Import DistributedSampler: `from torch.utils.data.distributed import DistributedSampler`
2. Add function `get_distributed_dataloader(dataset, batch_size, num_workers, rank, world_size)` that:
   - Creates DistributedSampler with num_replicas=world_size, rank=rank, shuffle=True, seed=42, drop_last=False
   - Returns DataLoader with sampler=sampler, pin_memory=True, persistent_workers=num_workers > 0
3. Modify main() to:
   - Pass rank and world_size to get_distributed_dataloader
   - Create both train and test dataloaders using this function

IMPORTANT: Do NOT use shuffle=True in DataLoader when using DistributedSampler - always use the sampler.
  </action>
  <verify>
<automated>grep -q "DistributedSampler" scripts/train_distributed.py && grep -q "sampler=sampler" scripts/train_distributed.py && echo "DistributedSampler integration found"</automated>
  </verify>
  <done>DistributedSampler integrated into train and test dataloaders with proper configuration</done>
</task>

<task type="auto">
  <name>Task 2: Add epoch synchronization and rank-aware logging</name>
  <files>scripts/train_distributed.py</files>
  <action>
Update scripts/train_distributed.py to add epoch synchronization and rank-aware logging:

1. Add epoch synchronization:
   - Before the training loop starts, create sampler = train_loader.sampler
   - At the start of each epoch: `sampler.set_epoch(epoch)`
   - After each epoch: call `barrier()` to synchronize all processes

2. Add rank-aware logging:
   - Replace all print() calls with log_on_main() (imported from distributed module)
   - Add proper logging in training loop: epoch progress, loss, accuracy
   - Only rank 0 should handle checkpoint saving (covered in Phase 2)

3. Ensure all print statements are guarded with is_main_process() check:
   - Startup messages
   - Epoch progress
   - Test accuracy
   - Summary statistics
  </action>
  <verify>
<automated>grep -q "sampler.set_epoch" scripts/train_distributed.py && grep -q "barrier()" scripts/train_distributed.py && echo "Epoch sync and barrier found"</automated>
  </verify>
  <done>Epoch synchronization implemented with set_epoch and barrier, all logging is rank-aware</done>
</task>

</tasks>

<verification>
- [ ] DistributedSampler integrated into train and test dataloaders
- [ ] sampler.set_epoch() called at start of each epoch
- [ ] barrier() called after each epoch for synchronization
- [ ] All print/logging calls use rank-aware functions
- [ ] No duplicate output when running with multiple GPUs
</verification>

<success_criteria>
Training script properly partitions data across GPUs, synchronizes epochs, and logs only from rank 0.
</success_criteria>

<output>
After completion, create `.planning/phases/01-distributed-infrastructure/01-distributed-infrastructure-02-SUMMARY.md`
</output>
