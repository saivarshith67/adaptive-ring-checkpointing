"""
Distributed training utilities for multi-GPU coordination.

Provides functions for:
- Process group initialization and cleanup
- Rank detection (global rank, local rank, world size)
- Process synchronization (barrier)
- Rank-aware logging (main process only)

These utilities work with torchrun for proper DDP process launch.
"""

import os
import torch
import torch.distributed as dist


# Global state to track initialization
_process_group_initialized = False
_default_pg = None


def is_distributed_initialized():
    """
    Check if the default process group is initialized.

    Returns:
        bool: True if distributed is initialized, False otherwise.
    """
    global _process_group_initialized, _default_pg
    return _process_group_initialized and _default_pg is not None


def get_rank():
    """
    Get the rank of the current process in the distributed group.

    Returns:
        int: Rank of the current process (0 if not distributed).
    """
    if not is_distributed_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """
    Get the total number of processes in the distributed group.

    Returns:
        int: World size (1 if not distributed).
    """
    if not is_distributed_initialized():
        return 1
    return dist.get_world_size()


def get_local_rank():
    """
    Get the local rank of the current process.

    The local rank is the GPU device index local to each node.
    This is set by torchrun via the LOCAL_RANK environment variable.

    Returns:
        int: Local rank (0 if not distributed).
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process():
    """
    Check if the current process is the main process (rank 0).

    Returns:
        bool: True if this is the main process, False otherwise.
    """
    return get_rank() == 0


def barrier():
    """
    Synchronize all processes in the distributed group.

    This blocks until all processes have reached this point.
    Useful for ensuring all processes are ready before proceeding.

    Raises:
        RuntimeError: If distributed is not initialized.
    """
    if is_distributed_initialized():
        dist.barrier()


def setup_distributed(backend="nccl"):
    """
    Initialize the distributed process group using torchrun environment variables.

    This function reads the following environment variables set by torchrun:
    - LOCAL_RANK: The local rank (GPU device index)
    - RANK: The global rank
    - WORLD_SIZE: The total number of processes

    Args:
        backend (str): The backend to use for communication. Default is "nccl".

    Returns:
        tuple: (rank, world_size, local_rank)

    Raises:
        RuntimeError: If required environment variables are missing or
                      initialization fails.
    """
    global _process_group_initialized, _default_pg

    # Check for torchrun environment variables
    rank = os.environ.get("RANK")
    world_size = os.environ.get("WORLD_SIZE")
    local_rank = os.environ.get("LOCAL_RANK")

    if rank is None or world_size is None:
        # Not launched with torchrun - treat as single-process mode
        _process_group_initialized = False
        _default_pg = None
        return 0, 1, 0

    rank = int(rank)
    world_size = int(world_size)
    local_rank = int(local_rank)

    # Initialize the process group
    init_method = "env://"

    # Set CUDA device before initialization
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )

    _default_pg = dist.group.WORLD
    _process_group_initialized = True

    return rank, world_size, local_rank


def cleanup_distributed():
    """
    Clean up the distributed process group.

    This should be called at the end of training to properly
    destroy the process group and release resources.

    Raises:
        RuntimeError: If distributed is not initialized.
    """
    global _process_group_initialized, _default_pg

    if is_distributed_initialized():
        dist.destroy_process_group()
        _process_group_initialized = False
        _default_pg = None


def log_on_main(*args, **kwargs):
    """
    Log message only on the main process (rank 0).

    This prevents duplicate output when running with multiple GPUs.

    Args:
        *args: Arguments passed to print().
        **kwargs: Keyword arguments passed to print().

    Example:
        >>> log_on_main("Training started")
        >>> log_on_main("Epoch %d/%d", epoch, total_epochs)
    """
    if is_main_process():
        print(*args, **kwargs)
