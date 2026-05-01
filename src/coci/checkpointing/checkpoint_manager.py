import torch
import torch.distributed as dist
import os
import time


class CheckpointManager:
    def __init__(self, checkpoint_dir="checkpoints", is_ddp_wrapped=False):
        self.checkpoint_dir = checkpoint_dir
        self.is_ddp_wrapped = is_ddp_wrapped
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 🔥 Metrics
        self.num_checkpoints = 0
        self.total_checkpoint_time = 0.0
        self.last_save_metadata = None
        self.last_load_metadata = None

    def _resolve_checkpoint_path(self, epoch, checkpoint_id=None):
        if checkpoint_id is None:
            filename = f"checkpoint_epoch_{epoch}.pt"
        else:
            filename = f"checkpoint_{checkpoint_id}.pt"
        return os.path.join(self.checkpoint_dir, filename)

    def save(
        self,
        model,
        optimizer,
        epoch,
        loss,
        metric=None,
        checkpoint_id=None,
    ):
        """
        Save a checkpoint.

        Args:
            model: The model to save (or DDP-wrapped model).
            optimizer: The optimizer to save.
            epoch: Current epoch number.
            loss: Current loss value.
            metric: Optional metric value (e.g., accuracy) to save.
        """
        # Only rank 0 saves checkpoint to avoid file conflicts
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank != 0:
            # Wait for rank 0 to finish saving
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            return

        path = self._resolve_checkpoint_path(epoch, checkpoint_id=checkpoint_id)

        start = time.time()

        # Use model.module.state_dict() if DDP wrapped, else model.state_dict()
        # This handles the extra wrapper layer that DDP adds
        state_dict = (
            model.module.state_dict() if self.is_ddp_wrapped else model.state_dict()
        )

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "checkpoint_id": checkpoint_id,
            "save_time": time.time(),
        }

        if metric is not None:
            checkpoint_data["metric"] = metric

        torch.save(checkpoint_data, path)

        duration = time.time() - start

        self.num_checkpoints += 1
        self.total_checkpoint_time += duration
        self.last_save_metadata = {
            "path": path,
            "epoch": epoch,
            "checkpoint_id": checkpoint_id,
            "metric": metric,
            "loss": loss,
            "duration_sec": duration,
            "load_source": "central-storage",
        }

        label = checkpoint_id if checkpoint_id is not None else f"epoch_{epoch}"
        print(f"[Checkpoint] Saved {label} | Time: {duration:.4f}s")

        # Barrier to ensure all ranks wait for save to complete
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def load_latest(self, model, optimizer, device, checkpoint_injector=None):
        """
        Load the latest checkpoint and restore model/optimizer state.

        Args:
            model: The model to load state into (or DDP-wrapped model).
            optimizer: The optimizer to load state into.
            device: Device for mapping checkpoint location.

        Returns:
            Tuple of (start_epoch, best_metric) where start_epoch is the next epoch
            to train and best_metric is the best validation metric from checkpoint.
        """
        files = [
            f
            for f in os.listdir(self.checkpoint_dir)
            if f.startswith("checkpoint_") and f.endswith(".pt")
        ]

        if not files:
            print("[Checkpoint] No checkpoint found. Starting fresh.")
            return 0, 0.0

        latest = max(
            files,
            key=lambda x: os.path.getmtime(os.path.join(self.checkpoint_dir, x)),
        )
        path = os.path.join(self.checkpoint_dir, latest)

        # Use map_location for device migration (GPU -> CPU or different GPU)
        checkpoint = torch.load(path, map_location=device)

        if checkpoint_injector is not None:
            checkpoint = checkpoint_injector.inject(checkpoint, path)

        # Use model.module.load_state_dict() if DDP wrapped
        load_target = model.module if self.is_ddp_wrapped else model
        load_target.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        epoch = int(checkpoint.get("epoch", 0))
        best_metric = checkpoint.get("metric", 0.0)
        self.last_load_metadata = {
            "path": path,
            "epoch": epoch,
            "checkpoint_id": checkpoint.get("checkpoint_id"),
            "metric": best_metric,
            "load_source": "central-storage",
        }
        print(
            f"[Checkpoint] Resumed from {latest} (epoch {epoch}, best_metric {best_metric:.4f})"
        )

        # Return next epoch to train and best metric for proper checkpoint tracking
        # epoch from checkpoint is 1-indexed (epochs completed), which equals
        # the next epoch to run (0-indexed)
        return epoch, best_metric

    def get_runtime_metrics(self):
        """Return checkpoint-manager runtime metrics in a hash-ring-compatible shape."""
        return {
            "load_source": (
                self.last_load_metadata.get("load_source")
                if self.last_load_metadata
                else "central-storage"
            ),
            "local_cache_hits": 0,
            "local_cache_misses": 0,
            "local_cache_loads": 0,
            "central_storage_loads": 1 if self.last_load_metadata else 0,
            "cache_write_count": 0,
            "total_shards": 0,
            "cache_size_mb": 0.0,
            "max_cache_size_mb": 0.0,
            "shards_owned": 0,
            "orphaned_shards": 0,
            "reassigned_shards": 0,
            "recached_shards": 0,
            "shard_recovery_count": 0,
            "recovery_time_sec": 0.0,
        }
