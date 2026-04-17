---
phase: 05-multi-gpu-fault-injection
plan: 01
subsystem: fault-tolerance
tags: [fault-injection, multi-gpu, ddp, testing]

# Dependency graph
requires:
  - phase: 04-exception-handling
    provides: Exception handling infrastructure, emergency checkpoint saving
provides:
  - Rank-aware fault injection for simulating GPU/node failures
  - CLI configuration for fault injection targets and rates
affects: [06-fault-recovery, testing]

# Tech tracking
tech-stack:
  added: [FaultInjector class]
  patterns: [Poisson failure model, rank-aware injection]

key-files:
  created: []
  modified:
    - src/coci/fault/fault_injector.py
    - scripts/train_faceforensics.py

key-decisions:
  - "Using target_rank=None for backward compatibility (all ranks inject)"
  - "Poisson model for realistic failure timing"
  - "Periodic injection every 100 batches to allow training progress"

patterns-established:
  - "FaultInjector initialization before training loop"
  - "maybe_fail() called conditionally on batch index"

requirements-completed: [FLTI-01, FLTI-02, FLTI-03]

# Metrics
duration: 2.5min
completed: 2026-04-17
---

# Phase 5 Plan 1: Multi-GPU Fault Injection Summary

**Rank-aware FaultInjector with CLI integration for simulating GPU failures**

## Performance

- **Duration:** 2.5 min
- **Started:** 2026-04-17T14:39:02Z
- **Completed:** 2026-04-17T14:41:32Z
- **Tasks:** 2 completed
- **Files modified:** 2

## Accomplishments
- Modified FaultInjector to support rank-aware fault injection via `target_rank` parameter
- Integrated fault injection CLI arguments into training script
- Enabled targeted failure injection for testing distributed fault tolerance

## Task Commits

Each task was committed atomically:

1. **Task 1: Modify FaultInjector for rank-aware injection** - `3a9f534` (feat)
2. **Task 2: Integrate fault injection into training script** - `a2a2d4b` (feat)

**Plan metadata:** `a2a2d4b` (docs: complete plan)

## Files Created/Modified
- `src/coci/fault/fault_injector.py` - Added `target_rank` and `rank` parameters for rank-aware injection
- `scripts/train_faceforensics.py` - Added `--inject-fault`, `--inject-rank`, `--inject-rate` CLI arguments

## Decisions Made
- Using `target_rank=None` for backward compatibility (existing behavior)
- Periodic injection (every 100 batches) instead of every batch to allow training progress
- Error messages include rank for easier debugging

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Ready for Phase 6 (fault recovery) which will use the fault injection infrastructure
- FaultInjector can be tested with: `torchrun --nproc_per_node=2 scripts/train_faceforensics.py --inject-fault --inject-rank 0`

---
*Phase: 05-multi-gpu-fault-injection*
*Completed: 2026-04-17*
