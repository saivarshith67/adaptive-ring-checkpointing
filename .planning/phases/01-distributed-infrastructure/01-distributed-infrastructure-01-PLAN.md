---
phase: 01-distributed-infrastructure
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/coci/distributed.py
  - scripts/train_distributed.py
autonomous: true
requirements:
  - MGPU-01
  - MGPU-02
  - MGPU-03

must_haves:
  truths:
    - "Process group can be initialized with NCCL backend via torchrun"
    - "Each process correctly identifies its rank, local_rank, and world_size"
    - "Entry point is torchrun-compatible and handles cleanup properly"
  artifacts:
    - path: "src/coci/distributed.py"
      provides: "Distributed utilities (setup, cleanup, rank detection)"
      exports: ["setup_distributed", "cleanup_distributed", "get_rank", "get_world_size", "get_local_rank", "is_main_process", "barrier", "log_on_main"]
    - path: "scripts/train_distributed.py"
      provides: "torchrun entry point for distributed training"
      contains: "setup_distributed", "torchrun", "cleanup_distributed"
  key_links:
    - from: "scripts/train_distributed.py"
      to: "src/coci/distributed.py"
      via: "import"
      pattern: "from src.coci.distributed import"
---

<objective>
Initialize distributed training infrastructure with process group setup and torchrun entry point.

Purpose: Create the foundational multi-GPU infrastructure required for distributed training in Phase 1.
Output: distributed.py module with utilities, train_distributed.py entry point
</objective>

<context>
@scripts/train.py (reference for existing training patterns)
@src/coci/data_ingestor/cifar.py (existing dataset code)

# Phase 1 Requirements (from ROADMAP.md)
- MGPU-01: Initialize distributed process group for multi-GPU coordination
- MGPU-02: Add torchrun-based entry point for proper DDP process launch
- MGPU-03: Implement rank-aware code paths (rank 0 vs other ranks)
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create distributed.py utilities module</name>
  <files>src/coci/distributed.py</files>
  <action>
Create src/coci/distributed.py with the following exports and functions:

1. `is_distributed_initialized()` - Check if process group is initialized
2. `get_rank()` - Get current process rank (0 if not distributed)
3. `get_world_size()` - Get total number of processes (1 if not distributed)
4. `get_local_rank()` - Get local rank from LOCAL_RANK env var (default 0)
5. `is_main_process()` - Check if current process is rank 0
6. `barrier()` - Synchronize all processes
7. `setup_distributed(backend="nccl")` - Initialize NCCL process group using env vars from torchrun
8. `cleanup_distributed()` - Destroy process group
9. `log_on_main(*args, **kwargs)` - Print only on rank 0

Use the pattern from 01-RESEARCH.md Pattern 1 and "Wrapper Function for Distributed Setup".
Ensure torch.cuda.set_device(local_rank) is called in setup_distributed.
  </action>
  <verify>
<automated>python -c "from src.coci.distributed import get_rank, get_world_size, is_main_process; print('Module loads OK')"</automated>
  </verify>
  <done>Module created with all required utilities, can be imported without errors</done>
</task>

<task type="auto">
  <name>Task 2: Create torchrun-based distributed entry point</name>
  <files>scripts/train_distributed.py</files>
  <action>
Create scripts/train_distributed.py that:

1. Imports from src.coci.distributed: setup_distributed, cleanup_distributed, is_main_process, get_rank, get_local_rank, get_world_size, barrier, log_on_main
2. Calls setup_distributed() at the start using torchrun environment variables (LOCAL_RANK, RANK, WORLD_SIZE)
3. Uses torchrun-compatible __main__ guard (if __name__ == "__main__":)
4. Sets CUDA device using torch.cuda.set_device(local_rank)
5. Prints startup message only on rank 0 (using is_main_process())
6. Contains minimal training loop stub showing the structure
7. Calls cleanup_distributed() in finally block

The script should work when launched with:
  torchrun --nproc_per_node=2 scripts/train_distributed.py
  </action>
  <verify>
<automated>python -c "import ast; ast.parse(open('scripts/train_distributed.py').read()); print('Syntax OK')"</automated>
  </verify>
  <done>Entry point created, parseable as valid Python, ready for integration with existing training code</done>
</task>

</tasks>

<verification>
- [ ] Module src/coci/distributed.py created with all required utilities
- [ ] scripts/train_distributed.py created with torchrun-compatible entry point
- [ ] Both files can be imported without errors
- [ ] train_distributed.py structure follows torchrun best practices
</verification>

<success_criteria>
Distributed infrastructure module loads successfully. Entry point is torchrun-compatible.
</success_criteria>

<output>
After completion, create `.planning/phases/01-distributed-infrastructure/01-distributed-infrastructure-01-SUMMARY.md`
</output>
