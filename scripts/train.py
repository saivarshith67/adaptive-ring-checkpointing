import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.coci.models.model import get_model
from src.coci.data_ingestor.cifar import get_cifar100_dataset
from src.coci.config import load_config
from src.coci.checkpointing.checkpoint_manager import CheckpointManager
from src.coci.fault.fault_injector import FaultInjector


# -------------------------------------------------
# Training for One Epoch
# -------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, fault_injector=None, inject_fault=False):
    model.train()
    total_loss = 0

    for batch_idx, (images, labels) in enumerate(loader):

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 🔥 Poisson mid-epoch failure
        if inject_fault and fault_injector is not None:
            fault_injector.maybe_fail()

    return total_loss


# -------------------------------------------------
# Evaluation
# -------------------------------------------------
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

    accuracy = 100 * correct / total
    return accuracy


# -------------------------------------------------
# Main
# -------------------------------------------------
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
        help="Enable Poisson fault injection"
    )

    args = parser.parse_args()

    # -------------------------
    # Config
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
    # Model & Optimizer
    # -------------------------
    model = get_model(cfg.model, num_classes=100)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # -------------------------
    # Checkpoint + Fault
    # -------------------------
    checkpoint_manager = CheckpointManager()
    start_epoch = checkpoint_manager.load_latest(model, optimizer)

    fault_injector = None
    if args.inject_fault:
        fault_injector = FaultInjector(
            failure_rate_per_second=cfg.failure_rate_per_second
        )

    # -------------------------
    # Training Loop
    # -------------------------
    try:
        for epoch in range(start_epoch, cfg.epochs):

            loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                fault_injector=fault_injector,
                inject_fault=args.inject_fault
            )

            print(f"Epoch {epoch+1}/{cfg.epochs}, Loss: {loss:.4f}")

            accuracy = evaluate(model, test_loader, device)
            print(f"Test Accuracy: {accuracy:.2f}%")

            checkpoint_manager.save(model, optimizer, epoch, loss)

    except RuntimeError as e:
        print(f"\n💥 Training interrupted due to failure: {e}")

    # -------------------------
    # MTBF Report
    # -------------------------
    if args.inject_fault and fault_injector is not None:
        total_time, failures = fault_injector.get_stats()

        if failures > 0:
            mtbf = total_time / failures
            print(f"\nTotal Runtime: {total_time:.2f}s")
            print(f"Failures: {failures}")
            print(f"Measured MTBF: {mtbf:.2f}s")
        else:
            print("\nNo failures occurred.")


if __name__ == "__main__":
    main()