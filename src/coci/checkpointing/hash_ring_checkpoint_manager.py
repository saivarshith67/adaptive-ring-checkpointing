import os
from typing import List, Optional, Tuple

import torch

from .checkpoint_manager import CheckpointManager
from ..hashing import create_hash_ring, create_shard_manager


class HashRingCheckpointManager:
    """Checkpoint manager that layers hash-ring local shard caching over normal checkpoints.

    This keeps central checkpoint files as the source of truth while opportunistically
    caching ring-owned checkpoints on local storage.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        is_ddp_wrapped: bool,
        node_id: str,
        all_node_ids: List[str],
        cache_root: str,
        virtual_nodes: int = 100,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.node_id = node_id
        self.all_node_ids = all_node_ids
        self.cache_root = cache_root

        self.base_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            is_ddp_wrapped=is_ddp_wrapped,
        )

        self.hash_ring = create_hash_ring(all_node_ids, virtual_nodes=virtual_nodes)
        node_cache_dir = os.path.join(cache_root, f"node_{node_id}")
        self.shard_manager = create_shard_manager(
            hash_ring=self.hash_ring,
            cache_dir=node_cache_dir,
        )

        os.makedirs(self.cache_root, exist_ok=True)

    def _checkpoint_filename(self, epoch: int) -> str:
        return f"checkpoint_epoch_{epoch}.pt"

    def _checkpoint_path(self, epoch: int) -> str:
        return os.path.join(self.checkpoint_dir, self._checkpoint_filename(epoch))

    def _shard_id(self, epoch: int) -> str:
        return f"checkpoint_epoch_{epoch}"

    def _register_shard_if_needed(self, epoch: int) -> None:
        shard_id = self._shard_id(epoch)
        if self.shard_manager.get_shard(shard_id) is not None:
            return

        path = self._checkpoint_path(epoch)
        size_bytes = os.path.getsize(path) if os.path.exists(path) else 0
        self.shard_manager.register_shard(
            shard_id=shard_id,
            epoch=epoch,
            size_bytes=size_bytes,
            central_path=path,
        )

    def _restore(self, checkpoint: dict, model, optimizer) -> Tuple[int, float]:
        load_target = model.module if self.base_manager.is_ddp_wrapped else model
        load_target.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        best_metric = checkpoint.get("metric", 0.0)
        return epoch + 1, best_metric

    def save(self, model, optimizer, epoch: int, loss: float, metric: Optional[float] = None):
        # Keep canonical checkpoint behavior.
        self.base_manager.save(model, optimizer, epoch, loss, metric=metric)

        self._register_shard_if_needed(epoch)
        shard_id = self._shard_id(epoch)
        owner = self.shard_manager.get_owner(shard_id)

        path = self._checkpoint_path(epoch)
        if owner != self.node_id or not os.path.exists(path):
            return

        try:
            state_dict = torch.load(path, map_location="cpu")
            cached = self.shard_manager.cache_shard(shard_id, state_dict)
            if cached:
                self.shard_manager.verify_shard(shard_id)
        except Exception:
            # Central checkpoint remains authoritative.
            pass

    def load_latest(self, model, optimizer, device):
        files = [
            f for f in os.listdir(self.checkpoint_dir) if f.startswith("checkpoint_epoch_")
        ]
        if not files:
            return 0, 0.0

        latest = max(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        epoch = int(latest.split("_")[-1].split(".")[0])

        self._register_shard_if_needed(epoch)
        shard_id = self._shard_id(epoch)

        if self.shard_manager.get_owner(shard_id) == self.node_id:
            cached = self.shard_manager.load_cached_shard(shard_id)
            if cached is not None:
                return self._restore(cached, model, optimizer)

        # Fallback path if local ring cache is not available.
        return self.base_manager.load_latest(model, optimizer, device)
