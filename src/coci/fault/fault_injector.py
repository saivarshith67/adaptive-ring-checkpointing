import random
import time
import math


class FaultInjector:
    """
    Rank-aware fault injector for simulating GPU/node failures.

    Supports two modes:
    - Global injection (target_rank=None): All ranks may inject failures
    - Targeted injection (target_rank=int): Only the specified rank injects failures

    Args:
        failure_rate_per_second: Poisson failure rate (lambda)
        target_rank: Specific rank to target (None = inject on all ranks)
        rank: Current process rank (used when target_rank is specified)
    """

    def __init__(self, failure_rate_per_second=0.02, target_rank=None, rank=0):
        self.lambda_rate = failure_rate_per_second
        self.target_rank = target_rank
        self.rank = rank
        self.last_check = time.time()
        self.start_time = time.time()
        self.failure_count = 0

    def maybe_fail(self):
        # Check if this rank should inject failure
        if self.target_rank is not None and self.rank != self.target_rank:
            return  # Skip injection for non-target ranks

        now = time.time()
        dt = now - self.last_check
        self.last_check = now

        # Poisson failure probability over dt
        prob = 1 - math.exp(-self.lambda_rate * dt)

        if random.random() < prob:
            self.failure_count += 1
            print(f"💥 Poisson failure triggered on rank {self.rank}!")
            raise RuntimeError(f"Injected Failure on rank {self.rank}")

    def get_stats(self):
        total_time = time.time() - self.start_time
        return total_time, self.failure_count
