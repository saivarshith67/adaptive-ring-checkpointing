#!/bin/bash

export TRAINING_MODE="pytorch-lightning"
export TRAIN_SCRIPT="scripts/train_pytorch_lightning.py"
export MODE_LABEL="PyTorch Lightning ModelCheckpoint"
export CHECKPOINT_SUFFIX="pytorch_lightning"

bash scripts/run_faceforensics_framework_backend.sh
