import time
from abc import ABC, abstractmethod
import math

class CheckpointStrategy(ABC):

    def __init__(self):
        self.last_checkpoint_time = time.time()

    @abstractmethod
    def should_checkpoint(self, **kwargs) -> bool:
        pass

    def update_checkpoint_time(self):
        self.last_checkpoint_time = time.time()
        
class FixedIntervalStrategy(CheckpointStrategy):

    def __init__(self, interval_seconds):
        super().__init__()
        self.interval = interval_seconds

    def should_checkpoint(self, **kwargs):
        now = time.time()
        if now - self.last_checkpoint_time >= self.interval:
            return True
        return False
        
        
class YoungDalyStrategy(CheckpointStrategy):

    def __init__(self, checkpoint_cost, mtbf):
        super().__init__()
        self.C = checkpoint_cost
        self.M = mtbf
        self.optimal_interval = math.sqrt(2 * self.C * self.M)

        print(f"[YoungDaly] Optimal interval: {self.optimal_interval:.2f} sec")

    def should_checkpoint(self, **kwargs):
        now = time.time()
        if now - self.last_checkpoint_time >= self.optimal_interval:
            return True
        return False
        
        
class EpochStrategy(CheckpointStrategy):

    def should_checkpoint(self, **kwargs):
        return True 


class FrameworkEpochStrategy(EpochStrategy):
    """Epoch checkpoint trigger for framework-native checkpoint backends."""

    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        
        
class CheckpointStrategyFactory:

    FRAMEWORK_STRATEGIES = {
        "pytorch-lightning",
        "hf-trainer",
        "deepspeed",
        "fsdp",
        "wandb-artifacts",
    }

    @staticmethod
    def create(cfg, checkpoint_cost=None, mtbf=None):

        if cfg.strategy == "fixed":
            return FixedIntervalStrategy(cfg.fixed_interval)

        elif cfg.strategy == "young_daly":
            return YoungDalyStrategy(checkpoint_cost, mtbf)

        elif cfg.strategy == "epoch":
            return EpochStrategy()

        elif cfg.strategy in CheckpointStrategyFactory.FRAMEWORK_STRATEGIES:
            return FrameworkEpochStrategy(cfg.strategy)

        else:
            raise ValueError("Unknown checkpoint strategy")
