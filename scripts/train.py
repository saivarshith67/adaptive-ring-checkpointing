import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time

from src.coci.models.model import get_model
from src.coci.data_ingestor.cifar import get_cifar100_dataset
from src.coci.config import load_config
from src.coci.checkpointing.checkpoint_manager import CheckpointManager
from src.coci.checkpointing.strategy import CheckpointStrategyFactory
from src.coci.fault.fault_injector import FaultInjector


# -------------------------------------------------
# Training Loop (Batch-Level Control)
# -------------------------------------------------
def train(
    model,
    train_loader,
    test_loader,
    optimizer,
    device,
    cfg,
    checkpoint_manager,
    strategy,
    fault_injector=None,
    inject_fault=False,
    start_epoch=0
):
    for epoch in range(start_epoch, cfg.epochs):

        model.train()
        total_loss = 0

        for batch_idx, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 🔥 Strategy-based checkpointing
            if strategy.should_checkpoint():
                checkpoint_time = checkpoint_manager.save(
                    model, optimizer, epoch, total_loss
                )
                strategy.update_checkpoint_time()

            # 🔥 Poisson failure
            if inject_fault and fault_injector is not None:
                fault_injector.maybe_fail()

        print(f"Epoch {epoch+1}/{cfg.epochs}, Loss: {total_loss:.4f}")

        accuracy = evaluate(model, test_loader, device)
        print(f"Test Accuracy: {accuracy:.2f}%")

        # If epoch-based strategy, checkpoint here
        if cfg.strategy == "epoch":
            checkpoint_manager.save(model, optimizer, epoch, total_loss)


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

    return 100 * correct / total


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dev", "server"], default="dev")
    parser.add_argument("--inject_fault", action="store_true")

    args = parser.parse_args()

    config_path = "configs/dev.yaml" if args.mode == "dev" else "configs/server.yaml"
    cfg = load_config(config_path)

    print(f"\nRunning in {args.mode.upper()} mode")
    print(f"Using strategy: {cfg.strategy}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # -------------------------
    # Dataset
    # -------------------------
    train_loader = DataLoader(
        get_cifar100_dataset(train=True),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    test_loader = DataLoader(
        get_cifar100_dataset(train=False),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    # -------------------------
    # Model
    # -------------------------
    model = get_model(cfg.model, num_classes=100)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # -------------------------
    # Checkpoint
    # -------------------------
    checkpoint_manager = CheckpointManager()
    start_epoch = checkpoint_manager.load_latest(model, optimizer)

    # -------------------------
    # Fault Injection
    # -------------------------
    fault_injector = None
    if args.inject_fault:
        fault_injector = FaultInjector(
            failure_rate_per_second=cfg.failure_rate_per_second
        )

    # -------------------------
    # Strategy Setup
    # -------------------------

    # For Young/Daly, we need checkpoint cost & MTBF
    # You can initially estimate these manually
    checkpoint_cost = cfg.checkpoint_cost_estimate
    mtbf_estimate = 1.0 / cfg.failure_rate_per_second if cfg.failure_rate_per_second > 0 else 1e9

    strategy = CheckpointStrategyFactory.create(
        cfg,
        checkpoint_cost=checkpoint_cost,
        mtbf=mtbf_estimate
    )

    # -------------------------
    # Train
    # -------------------------
    try:
        train(
            model,
            train_loader,
            test_loader,
            optimizer,
            device,
            cfg,
            checkpoint_manager,
            strategy,
            fault_injector=fault_injector,
            inject_fault=args.inject_fault,
            start_epoch=start_epoch
        )

    except RuntimeError as e:
        print(f"\n💥 Training interrupted: {e}")

    # -------------------------
    # MTBF Report
    # -------------------------
    if args.inject_fault and fault_injector is not None:
        total_time, failures = fault_injector.get_stats()
        if failures > 0:
            print(f"\nTotal Runtime: {total_time:.2f}s")
            print(f"Failures: {failures}")
            print(f"Measured MTBF: {total_time / failures:.2f}s")
        else:
            print("\nNo failures occurred.")


if __name__ == "__main__":
    main()