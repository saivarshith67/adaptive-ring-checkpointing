from .hash_ring import HashRing, ShardManager
from .fault_detector import FaultDetector, ElasticRecaching, NodeStatus
from .recovery_scheduler import RecoveryScheduler, ResumeMode


__all__ = [
    "HashRing",
    "ShardManager",
    "FaultDetector",
    "ElasticRecaching",
    "NodeStatus",
    "RecoveryScheduler",
    "ResumeMode",
]


def create_hash_ring(node_ids: list, virtual_nodes: int = 100) -> HashRing:
    ring = HashRing(virtual_nodes_per_physical=virtual_nodes)
    for node_id in node_ids:
        ring.add_node(node_id)
    return ring


def register_fault_detector_nodes(
    fault_detector: FaultDetector,
    hash_ring: HashRing,
) -> None:
    """Register all nodes from a HashRing with a FaultDetector.

    Args:
        fault_detector: The FaultDetector instance
        hash_ring: The HashRing containing nodes to register
    """
    for node_id in hash_ring.get_nodes():
        fault_detector.register_node(node_id)


def create_shard_manager(
    hash_ring: HashRing,
    cache_dir: str = "/nvme/cache",
) -> ShardManager:
    return ShardManager(hash_ring=hash_ring, cache_dir=cache_dir)


def create_fault_detector(
    ttl_seconds: float = 5.0,
    timeout_limit: int = 3,
    heartbeat_interval: float = 1.0,
    suspicion_quorum_pct: int = 51,
) -> FaultDetector:
    return FaultDetector(
        ttl_seconds=ttl_seconds,
        timeout_limit=timeout_limit,
        heartbeat_interval=heartbeat_interval,
        suspicion_quorum_pct=suspicion_quorum_pct,
    )


def create_recovery_scheduler(
    hot_resume_threshold: float = 0.80,
    critical_failure_node_loss_pct: float = 0.20,
) -> RecoveryScheduler:
    return RecoveryScheduler(
        hot_resume_threshold=hot_resume_threshold,
        critical_failure_node_loss_pct=critical_failure_node_loss_pct,
    )
