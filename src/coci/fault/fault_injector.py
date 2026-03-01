import random
import time


class FaultInjector:
    def __init__(self, failure_probability=0.2):
        self.failure_probability = failure_probability
        self.failure_count = 0
        self.start_time = time.time()

    def maybe_fail(self):
        if random.random() < self.failure_probability:
            self.failure_count += 1
            print("💥 Simulated Failure Triggered!")
            raise RuntimeError("Injected Failure")

    def get_stats(self):
        total_time = time.time() - self.start_time
        return total_time, self.failure_count