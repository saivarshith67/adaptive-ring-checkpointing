import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.coci.models.model import get_model
from src.coci.data.cifar import get_cifar100_dataset
from src.coci.config import load_config
from src.coci.checkpointing.checkpoint_manager import CheckpointManager
from src.coci.fault.fault_injector import FaultInjector


def main():

    # -------------------------
    # Argument parser
    # -------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["dev", "server"],
        default="dev",
        help="Run mode: dev (local) or server (HPC)"
    )
    parser.add_argument(
        "--inject_fault",
        action="store_true",
        help="Enable fault injection"
    )

    args = parser.parse_args()

    # -------------------------
    # Config selection
    # -------------------------
    config_path = "configs/dev.yaml" if args.mode == "dev" else "configs/server.yaml"

    print(f"\nRunning in {args.mode.upper()} mode")
    print(f"Loading config: {config_path}")

    cfg = load_config(config_path)

    # -------------------------
    # Device
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # -------------------------
    # Dataset
    # -------------------------
    train_dataset = get_cifar100_dataset(train=True)
    test_dataset = get_cifar100_dataset(train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    # -------------------------
    # Model
    # -------------------------
    model = get_model(cfg.model, num_classes=100)
    model.to(device)
    failure_probability = cfg.failure_prob

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # -------------------------
    # Checkpoint + Fault Setup
    # -------------------------
    checkpoint_manager = CheckpointManager()
    start_epoch = checkpoint_manager.load_latest(model, optimizer)

    fault_injector = FaultInjector(failure_probability=failure_probability)

    # -------------------------
    # Training loop
    # -------------------------
    try:
        for epoch in range(start_epoch, cfg.epochs):

            model.train()
            total_loss = 0

            for images, labels in train_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{cfg.epochs}, Loss: {total_loss:.4f}")

            evaluate(model, test_loader, device)

            # Save checkpoint every epoch
            checkpoint_manager.save(model, optimizer, epoch, total_loss)

            # Fault injection
            if args.inject_fault:
                fault_injector.maybe_fail()

    except RuntimeError as e:
        print(f"Training interrupted due to failure: {e}")

    # Print MTBF stats
    if args.inject_fault:
        total_time, failures = fault_injector.get_stats()
        if failures > 0:
            mtbf = total_time / failures
            print(f"\nTotal Runtime: {total_time:.2f}s")
            print(f"Failures: {failures}")
            print(f"Estimated MTBF: {mtbf:.2f}s")
        else:
            print("\nNo failures occurred.")


def evaluate(model, loader, device):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()