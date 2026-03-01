import os
import torch
import time


class CheckpointManager:
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self, model, optimizer, epoch, loss):
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")

        start = time.time()

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }, path)

        checkpoint_time = time.time() - start
        print(f"[Checkpoint] Saved epoch {epoch} | Time: {checkpoint_time:.4f}s")

        return checkpoint_time

    def load_latest(self, model, optimizer):
        files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pt")]

        if not files:
            print("[Checkpoint] No checkpoint found. Starting fresh.")
            return 0

        latest = sorted(files)[-1]
        path = os.path.join(self.checkpoint_dir, latest)

        checkpoint = torch.load(path)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1

        print(f"[Checkpoint] Resumed from {latest}")

        return start_epoch