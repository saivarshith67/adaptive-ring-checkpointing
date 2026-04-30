#!/bin/bash

export TRAINING_MODE="wandb-artifacts"
export TRAIN_SCRIPT="scripts/train_wandb_artifacts.py"
export MODE_LABEL="Weights & Biases Artifacts"
export CHECKPOINT_SUFFIX="wandb_artifacts"

bash scripts/run_faceforensics_framework_backend.sh
