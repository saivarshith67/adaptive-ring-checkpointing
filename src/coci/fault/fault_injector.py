import random
import time
import math


class FaultInjector:
    def __init__(self, failure_rate_per_second=0.02):
        self.lambda_rate = failure_rate_per_second
        self.last_check = time.time()
        self.start_time = time.time()
        self.failure_count = 0

    def maybe_fail(self):
        now = time.time()
        dt = now - self.last_check
        self.last_check = now

        # Poisson failure probability over dt
        prob = 1 - math.exp(-self.lambda_rate * dt)

        if random.random() < prob:
            self.failure_count += 1
            print("💥 Poisson failure triggered!")
            raise RuntimeError("Injected Failure")

    def get_stats(self):
        total_time = time.time() - self.start_time
        return total_time, self.failure_count