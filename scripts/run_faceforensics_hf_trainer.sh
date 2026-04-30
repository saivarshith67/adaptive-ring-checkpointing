#!/bin/bash

export TRAINING_MODE="hf-trainer"
export TRAIN_SCRIPT="scripts/train_hf_trainer.py"
export MODE_LABEL="Hugging Face Trainer Checkpointing"
export CHECKPOINT_SUFFIX="hf_trainer"

bash scripts/run_faceforensics_framework_backend.sh
