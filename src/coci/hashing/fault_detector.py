import time
import threading
from typing import Dict, Optional, List
from enum import Enum


class NodeStatus(Enum):
    ALIVE = "ALIVE"
    SUSPECTED = "SUSPECTED"
    DEAD = "DEAD"


class FaultDetector:
    def __init__(
        self,
        ttl_seconds: float = 5.0,
        timeout_limit: int = 3,
        heartbeat_interval: float = 1.0,
        suspicion_quorum_pct: int = 51,
    ):
        self.ttl_seconds = ttl_seconds
        self.timeout_limit = timeout_limit
        self.heartbeat_interval = heartbeat_interval
        self.suspicion_quorum_pct = suspicion_quorum_pct

        self.node_status: Dict[str, NodeStatus] = {}
        self.timeout_count: Dict[str, int] = {}
        self.last_heartbeat: Dict[str, float] = {}
        self.suspicion_votes: Dict[str, List[str]] = {}

        self._lock = threading.Lock()

    def register_node(self, node_id: str) -> None:
        with self._lock:
            self.node_status[node_id] = NodeStatus.ALIVE
            self.timeout_count[node_id] = 0
            self.last_heartbeat[node_id] = time.time()
            self.suspicion_votes[node_id] = []

    def mark_alive(self, node_id: str) -> None:
        with self._lock:
            if node_id not in self.node_status:
                self.register_node(node_id)
            self.node_status[node_id] = NodeStatus.ALIVE
            self.timeout_count[node_id] = 0
            self.last_heartbeat[node_id] = time.time()

    def register_suspicion(self, node_id: str, suspector_id: str) -> bool:
        with self._lock:
            if node_id not in self.suspicion_votes:
                self.suspicion_votes[node_id] = []

            if suspector_id not in self.suspicion_votes[node_id]:
                self.suspicion_votes[node_id].append(suspector_id)

            total_nodes = len(self.node_status)
            votes_needed = (total_nodes * self.suspicion_quorum_pct + 99) // 100

            if len(self.suspicion_votes[node_id]) >= votes_needed:
                self.node_status[node_id] = NodeStatus.DEAD
                return True

            self.node_status[node_id] = NodeStatus.SUSPECTED
            return False

    def check_timeout(self, node_id: str) -> None:
        with self._lock:
            if node_id not in self.last_heartbeat:
                return

            elapsed = time.time() - self.last_heartbeat[node_id]
            if elapsed > self.ttl_seconds:
                self.timeout_count[node_id] = self.timeout_count.get(node_id, 0) + 1

                if self.timeout_count[node_id] >= self.timeout_limit:
                    self.node_status[node_id] = NodeStatus.SUSPECTED

    def get_status(self, node_id: str) -> Optional[NodeStatus]:
        return self.node_status.get(node_id)

    def get_dead_nodes(self) -> List[str]:
        return [
            node_id
            for node_id, status in self.node_status.items()
            if status == NodeStatus.DEAD
        ]

    def get_suspected_nodes(self) -> List[str]:
        return [
            node_id
            for node_id, status in self.node_status.items()
            if status == NodeStatus.SUSPECTED
        ]

    def get_alive_nodes(self) -> List[str]:
        return [
            node_id
            for node_id, status in self.node_status.items()
            if status == NodeStatus.ALIVE
        ]


class ElasticRecaching:
    def __init__(self, shard_manager, fault_detector: FaultDetector):
        self.shard_manager = shard_manager
        self.fault_detector = fault_detector

    def handle_node_failure(self, dead_node: str) -> List[str]:
        orphaned = self.shard_manager.get_orphaned_shards(dead_node)
        for shard_id in orphaned:
            self.shard_manager.update_status(shard_id, "ORPHANED")

        return orphaned

    def recache_shard(self, shard_id: str) -> bool:
        shard = self.shard_manager.get_shard(shard_id)
        if not shard:
            return False

        if shard["status"] == "ORPHANED":
            self.shard_manager.update_status(shard_id, "CACHING")
            self.shard_manager.update_status(shard_id, "RE-CACHED")
            return True

        return False

    def get_recache_candidates(self, node_id: str) -> List[str]:
        owned = [
            sid
            for sid, info in self.shard_manager.shard_table.items()
            if info["owner_node"] == node_id
        ]
        return [
            sid
            for sid in owned
            if self.shard_manager.get_shard(sid)["status"] == "ORPHANED"
        ]
