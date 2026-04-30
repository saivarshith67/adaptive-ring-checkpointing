#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_faceforensics import main


if __name__ == "__main__":
    sys.exit(main(["--training-mode", "wandb-artifacts", *sys.argv[1:]]))
