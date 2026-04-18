from enum import Enum
from typing import List, Dict, Optional, Tuple


class ResumeMode(Enum):
    HOT_RESUME = "HOT_RESUME"
    COLD_RESUME = "COLD_RESUME"
    CRITICAL_FAILURE = "CRITICAL_FAILURE"


class RecoveryScheduler:
    def __init__(
        self,
        hot_resume_threshold: float = 0.80,
        critical_failure_node_loss_pct: float = 0.20,
    ):
        self.hot_resume_threshold = hot_resume_threshold
        self.critical_failure_node_loss_pct = critical_failure_node_loss_pct

        self.last_clean_epoch: int = 0
        self.victim_epoch: int = 0
        self.resume_from_epoch: int = 0

    def select_resume_mode(
        self, epoch_progress: float, nodes_lost: int, total_nodes: int
    ) -> ResumeMode:
        loss_ratio = nodes_lost / total_nodes if total_nodes > 0 else 0

        if loss_ratio > self.critical_failure_node_loss_pct:
            return ResumeMode.CRITICAL_FAILURE
        elif epoch_progress >= self.hot_resume_threshold:
            return ResumeMode.HOT_RESUME
        else:
            return ResumeMode.COLD_RESUME

    def compute_rollback(
        self,
        current_epoch: int,
        epoch_progress: float,
        nodes_lost: int,
        total_nodes: int,
    ) -> Tuple[int, int, ResumeMode]:
        mode = self.select_resume_mode(epoch_progress, nodes_lost, total_nodes)

        self.last_clean_epoch = current_epoch - 1 if current_epoch > 0 else 0
        self.victim_epoch = current_epoch

        if mode == ResumeMode.HOT_RESUME:
            self.resume_from_epoch = self.last_clean_epoch
        elif mode == ResumeMode.COLD_RESUME:
            self.resume_from_epoch = self.last_clean_epoch
        else:
            self.resume_from_epoch = self.last_clean_epoch

        return self.resume_from_epoch, current_epoch, mode

    def rebalance_work(
        self,
        surviving_nodes: List[str],
        total_shards: int,
        shard_to_node: Dict[str, str],
    ) -> Dict[str, List[str]]:
        if not surviving_nodes:
            return {}

        shards_per_node = total_shards // len(surviving_nodes)
        remainder = total_shards % len(surviving_nodes)

        work_slices: Dict[str, List[str]] = {node: [] for node in surviving_nodes}

        shard_list = list(shard_to_node.keys())
        idx = 0

        for node in surviving_nodes:
            count = shards_per_node + (1 if remainder > 0 else 0)
            remainder -= 1 if remainder > 0 else 0

            node_shards = []
            for _ in range(count):
                if idx < len(shard_list):
                    node_shards.append(shard_list[idx])
                    idx += 1

            work_slices[node] = node_shards

        return work_slices

    def create_manifest(
        self,
        epoch: int,
        model_state_path: str,
        optimizer_state_path: str,
        participants: List[str],
    ) -> Dict:
        return {
            "epoch": epoch,
            "status": "COMPLETE",
            "model_state_path": model_state_path,
            "optimizer_state_path": optimizer_state_path,
            "shard_manifest": {},
            "participants": participants,
        }
