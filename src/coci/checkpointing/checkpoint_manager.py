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

    def save(self, model, optimizer, epoch, loss, metric=None):
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

        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")

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
        }

        if metric is not None:
            checkpoint_data["metric"] = metric

        torch.save(checkpoint_data, path)

        duration = time.time() - start

        self.num_checkpoints += 1
        self.total_checkpoint_time += duration

        print(f"[Checkpoint] Saved epoch {epoch} | Time: {duration:.4f}s")

        # Barrier to ensure all ranks wait for save to complete
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def load_latest(self, model, optimizer, device):
        files = [
            f
            for f in os.listdir(self.checkpoint_dir)
            if f.startswith("checkpoint_epoch_")
        ]

        if not files:
            print("[Checkpoint] No checkpoint found. Starting fresh.")
            return 0

        latest = max(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        path = os.path.join(self.checkpoint_dir, latest)

        # Use map_location for device migration (GPU -> CPU or different GPU)
        checkpoint = torch.load(path, map_location=device)

        # Use model.module.load_state_dict() if DDP wrapped
        load_target = model.module if self.is_ddp_wrapped else model
        load_target.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        epoch = checkpoint["epoch"]
        print(f"[Checkpoint] Resumed from {latest}")

        return epoch + 1
