import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.coci.models.model import get_model
from src.coci.data.cifar import get_cifar100_dataset
from src.coci.config import load_config


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

    args = parser.parse_args()

    # -------------------------
    # Select config file
    # -------------------------
    if args.mode == "dev":
        config_path = "configs/dev.yaml"
    else:
        config_path = "configs/server.yaml"

    print(f"\nRunning in {args.mode.upper()} mode")
    print(f"Loading config: {config_path}")

    cfg = load_config(config_path)

    # -------------------------
    # Device auto-detection
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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(cfg.epochs):

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