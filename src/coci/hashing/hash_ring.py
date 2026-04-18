import hashlib
import bisect
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


class HashRing:
    def __init__(self, virtual_nodes_per_physical: int = 100):
        self.virtual_nodes_per_physical = virtual_nodes_per_physical
        self.ring: Dict[float, str] = {}
        self.virtual_nodes: Dict[str, List[float]] = {}
        self.sorted_positions: List[float] = []

    def _hash_position(self, key: str) -> float:
        hash_bytes = hashlib.sha256(key.encode()).digest()
        hash_int = int.from_bytes(hash_bytes[:8], byteorder="big")
        return hash_int / (2**64)

    def add_node(self, node_id: str) -> None:
        if node_id in self.virtual_nodes:
            return

        positions = []
        for vnode_idx in range(self.virtual_nodes_per_physical):
            vnode_key = f"{node_id}:{vnode_idx}"
            position = self._hash_position(vnode_key)
            positions.append(position)
            self.ring[position] = node_id

        self.virtual_nodes[node_id] = positions
        self.sorted_positions = sorted(self.ring.keys())

    def remove_node(self, node_id: str) -> None:
        if node_id not in self.virtual_nodes:
            return

        for position in self.virtual_nodes[node_id]:
            del self.ring[position]

        del self.virtual_nodes[node_id]
        self.sorted_positions = sorted(self.ring.keys())

    def get_node(self, key: str) -> Optional[str]:
        if not self.sorted_positions:
            return None

        position = self._hash_position(key)
        idx = bisect.bisect_right(self.sorted_positions, position)

        if idx >= len(self.sorted_positions):
            idx = 0

        return self.ring[self.sorted_positions[idx]]

    def get_nodes(self) -> List[str]:
        return list(self.virtual_nodes.keys())

    def get_virtual_node_count(self, node_id: str) -> int:
        return len(self.virtual_nodes.get(node_id, []))


class ShardManager:
    ShardStatus = [
        "UNCACHED",
        "CACHING",
        "CACHED",
        "ORPHANED",
        "RE-CACHED",
    ]

    def __init__(self, hash_ring: HashRing, cache_dir: str = "/nvme/cache"):
        self.hash_ring = hash_ring
        self.cache_dir = cache_dir
        self.shard_table: Dict[str, Dict] = {}

    def register_shard(
        self,
        shard_id: str,
        epoch: int,
        size_bytes: int,
        central_path: str,
    ) -> None:
        owner = self.hash_ring.get_node(shard_id) or self.hash_ring.get_nodes()[0]
        self.shard_table[shard_id] = {
            "shard_id": shard_id,
            "epoch": epoch,
            "size_bytes": size_bytes,
            "owner_node": owner,
            "status": "UNCACHED",
            "central_path": central_path,
            "cached_path": f"{self.cache_dir}/{shard_id}.pt",
            "last_verified": None,
        }

    def get_shard(self, shard_id: str) -> Optional[Dict]:
        return self.shard_table.get(shard_id)

    def get_owner(self, shard_id: str) -> Optional[str]:
        shard = self.shard_table.get(shard_id)
        return shard["owner_node"] if shard else None

    def update_status(self, shard_id: str, status: str) -> None:
        if shard_id in self.shard_table:
            self.shard_table[shard_id]["status"] = status

    def reassign_shard(self, shard_id: str) -> str:
        new_owner = self.hash_ring.get_node(shard_id)
        if shard_id in self.shard_table:
            old_owner = self.shard_table[shard_id]["owner_node"]
            if new_owner != old_owner:
                self.shard_table[shard_id]["owner_node"] = new_owner
                if self.shard_table[shard_id]["status"] == "CACHED":
                    self.shard_table[shard_id]["status"] = "ORPHANED"
        return new_owner

    def get_orphaned_shards(self, dead_node: str) -> List[str]:
        return [
            shard_id
            for shard_id, info in self.shard_table.items()
            if info["owner_node"] == dead_node
            and info["status"] in ["CACHED", "RE-CACHED"]
        ]

    def get_all_shards(self) -> List[str]:
        return list(self.shard_table.keys())

    def cache_shard(self, shard_id: str, state_dict: dict) -> bool:
        """Write shard to local NVMe cache with atomic rename pattern."""
        if shard_id not in self.shard_table:
            return False

        try:
            self.update_status(shard_id, "CACHING")

            cached_path = Path(self.cache_dir) / f"{shard_id}.pt"
            cached_path.parent.mkdir(parents=True, exist_ok=True)

            # Atomic write: temp file + fsync + move (close file first on Windows)
            tmp_fd, tmp_path = tempfile.mkstemp(suffix='.pt')
            try:
                with os.fdopen(tmp_fd, 'wb') as tmp_file:
                    torch.save(state_dict, tmp_file)
                    tmp_file.flush()
                    os.fsync(tmp_file.fileno())
                shutil.move(tmp_path, str(cached_path))
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise

            self.update_status(shard_id, "CACHED")
            return True

        except Exception:
            self.update_status(shard_id, "UNCACHED")
            return False

    def load_cached_shard(self, shard_id: str) -> Optional[dict]:
        """Load shard from local cache with verification."""
        if shard_id not in self.shard_table:
            return None

        cached_path = Path(self.cache_dir) / f"{shard_id}.pt"
        if not cached_path.exists():
            self.update_status(shard_id, "ORPHANED")
            return None

        try:
            state_dict = torch.load(cached_path, map_location='cpu')
            self.update_status(shard_id, "CACHED")
            return state_dict

        except Exception:
            return None

    def verify_shard(self, shard_id: str) -> bool:
        """Verify shard integrity by loading and checking."""
        cached_path = Path(self.cache_dir) / f"{shard_id}.pt"
        if not cached_path.exists():
            return False

        try:
            torch.load(cached_path, map_location='cpu')
            self.shard_table[shard_id]['last_verified'] = datetime.utcnow().isoformat()
            return True
        except Exception:
            return False

    def get_cache_usage(self) -> dict:
        """Return cache usage statistics."""
        cache_dir = Path(self.cache_dir)
        if not cache_dir.exists():
            return {'total_bytes': 0, 'file_count': 0, 'cache_dir': str(cache_dir)}

        files = list(cache_dir.glob('*.pt'))
        total_bytes = sum(f.stat().st_size for f in files if f.is_file())
        return {'total_bytes': total_bytes, 'file_count': len(files), 'cache_dir': str(cache_dir)}

    def cleanup_orphaned_files(self, dead_node: str) -> int:
        """Remove cached files for shards owned by a dead node."""
        orphaned = self.get_orphaned_shards(dead_node)
        removed_count = 0

        for shard_id in orphaned:
            cached_path = Path(self.cache_dir) / f"{shard_id}.pt"
            if cached_path.exists():
                cached_path.unlink()
                removed_count += 1

        return removed_count
