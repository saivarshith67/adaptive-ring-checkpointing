import torch
import os
import time

class CheckpointManager:
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 🔥 Metrics
        self.num_checkpoints = 0
        self.total_checkpoint_time = 0.0

    def save(self, model, optimizer, epoch, loss):
        path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_epoch_{epoch}.pt"
        )

        start = time.time()

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }, path)

        duration = time.time() - start

        self.num_checkpoints += 1
        self.total_checkpoint_time += duration

        print(f"[Checkpoint] Saved epoch {epoch} | Time: {duration:.4f}s")

    def load_latest(self, model, optimizer):
        files = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith("checkpoint_epoch_")
        ]

        if not files:
            print("[Checkpoint] No checkpoint found. Starting fresh.")
            return 0

        latest = max(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        path = os.path.join(self.checkpoint_dir, latest)

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        epoch = checkpoint["epoch"]
        print(f"[Checkpoint] Resumed from {latest}")

        return epoch + 1