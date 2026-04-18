import time
import threading
from typing import Dict, Optional, List
from enum import Enum

import torch
import torch.distributed as dist


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
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        self.ttl_seconds = ttl_seconds
        self.timeout_limit = timeout_limit
        self.heartbeat_interval = heartbeat_interval
        self.suspicion_quorum_pct = suspicion_quorum_pct
        self.rank = rank
        self.world_size = world_size

        self.node_status: Dict[str, NodeStatus] = {}
        self.timeout_count: Dict[str, int] = {}
        self.last_heartbeat: Dict[str, float] = {}
        self.suspicion_votes: Dict[str, List[str]] = {}

        # Heartbeat thread state
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None

        self._lock = threading.Lock()

    @property
    def is_distributed(self) -> bool:
        """Check if running in distributed mode."""
        try:
            return (
                dist.is_available()
                and dist.is_initialized()
                and self.rank is not None
                and self.world_size is not None
            )
        except Exception:
            return False

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

    def broadcast_heartbeat(self) -> Dict[str, int]:
        """Broadcast heartbeat signals to all peers using dist.all_reduce.

        Returns aggregated heartbeat counts from all nodes as {node_id: count}.
        In non-distributed mode, returns local heartbeats only.
        """
        if not self.is_distributed:
            # Non-distributed mode: return local heartbeats only
            local_heartbeats = {
                node_id: 1
                for node_id in self.node_status
                if self.node_status[node_id] == NodeStatus.ALIVE
            }
            return local_heartbeats

        try:
            # Create tensor with local heartbeat count per node
            local_counts = torch.zeros(len(self.node_status), dtype=torch.long)
            for idx, node_id in enumerate(self.node_status):
                if self.node_status[node_id] == NodeStatus.ALIVE:
                    local_counts[idx] = 1

            # All-reduce to aggregate across all ranks
            total_counts = [torch.zeros_like(local_counts) for _ in range(self.world_size)]
            dist.all_gather(total_counts, local_counts)

            # Aggregate results
            aggregated = {}
            for rank_idx, counts in enumerate(total_counts):
                for idx, node_id in enumerate(self.node_status):
                    aggregated[node_id] = aggregated.get(node_id, 0) + counts[idx].item()

            return aggregated

        except Exception:
            # Fallback to local-only on error
            return {
                node_id: 1
                for node_id in self.node_status
                if self.node_status[node_id] == NodeStatus.ALIVE
            }

    def gather_suspicions(self) -> Dict[str, int]:
        """Gather suspicion votes from all ranks using dist.all_gather.

        Returns aggregated suspicion votes per node as {node_id: vote_count}.
        In non-distributed mode, returns local votes only.
        """
        if not self.is_distributed:
            # Non-distributed mode: return local votes only
            local_votes = {
                node_id: len(voters)
                for node_id, voters in self.suspicion_votes.items()
            }
            return local_votes

        try:
            # Create tensor with local suspicion counts
            local_counts = torch.zeros(len(self.node_status), dtype=torch.long)
            for idx, node_id in enumerate(self.node_status):
                local_counts[idx] = len(self.suspicion_votes.get(node_id, []))

            # All-gather to collect from all ranks
            total_counts = [torch.zeros_like(local_counts) for _ in range(self.world_size)]
            dist.all_gather(total_counts, local_counts)

            # Aggregate results
            aggregated = {}
            for rank_idx, counts in enumerate(total_counts):
                for idx, node_id in enumerate(self.node_status):
                    aggregated[node_id] = aggregated.get(node_id, 0) + counts[idx].item()

            return aggregated

        except Exception:
            # Fallback to local-only on error
            return {
                node_id: len(voters)
                for node_id, voters in self.suspicion_votes.items()
            }

    def start_heartbeat_thread(self, rank: Optional[int] = None, world_size: Optional[int] = None) -> None:
        """Start background thread for periodic heartbeat broadcast.

        Args:
            rank: Current process rank (defaults to self.rank)
            world_size: Total number of processes (defaults to self.world_size)
        """
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return  # Already running

        self.rank = rank if rank is not None else self.rank
        self.world_size = world_size if world_size is not None else self.world_size

        self._stop_event = threading.Event()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
        )
        self._heartbeat_thread.start()

    def stop_heartbeat_thread(self, timeout: float = 2.0) -> None:
        """Stop the heartbeat thread gracefully.

        Args:
            timeout: Maximum seconds to wait for thread to finish
        """
        if self._stop_event is None:
            return

        self._stop_event.set()

        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=timeout)

        self._heartbeat_thread = None
        self._stop_event = None

    def _heartbeat_loop(self) -> None:
        """Background loop that periodically broadcasts heartbeats."""
        while not self._stop_event.is_set():
            try:
                # Broadcast local heartbeat
                self.broadcast_heartbeat()

                # Check for timeouts in other nodes
                for node_id in list(self.node_status.keys()):
                    self.check_timeout(node_id)

                    # Register suspicion if node timed out multiple times
                    status = self.node_status.get(node_id)
                    if status == NodeStatus.SUSPECTED:
                        self.register_suspicion(node_id, f"rank:{self.rank}")

            except Exception:
                pass  # Continue on errors

            # Wait for next interval or stop signal
            self._stop_event.wait(self.heartbeat_interval)

    def get_status_summary(self) -> Dict[str, int]:
        """Get a summary of node statuses.

        Returns:
            Dict with 'alive', 'suspected', 'dead' counts
        """
        with self._lock:
            return {
                "alive": sum(
                    1 for s in self.node_status.values() if s == NodeStatus.ALIVE
                ),
                "suspected": sum(
                    1 for s in self.node_status.values() if s == NodeStatus.SUSPECTED
                ),
                "dead": sum(
                    1 for s in self.node_status.values() if s == NodeStatus.DEAD
                ),
            }


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
