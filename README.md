# Adaptive Ring Checkpointing

Fault-tolerant distributed deep learning training with two complementary ideas:

- Convergence-aware checkpoint timing (COCI-inspired): decides when to checkpoint
- Hash-ring-based shard placement and recovery: decides where to checkpoint and how to recover

The project targets GPU-heavy HPC style training where failures are expensive and restart overhead must be minimized.

## Command Cheat Sheet

Use these as quick copy-paste commands.

### 1) Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Dataset Download

```bash
python scripts/train_faceforensics.py --download-dataset
```

### 3) Train - 4 Modes (Single GPU)

```bash
python scripts/train_epoch_based.py --dataset-path ./data/faceforensics/FF++
python scripts/train_convergence_normal.py --dataset-path ./data/faceforensics/FF++
python scripts/train_hashring_epoch.py --dataset-path ./data/faceforensics/FF++
python scripts/train_convergence_hashring.py --dataset-path ./data/faceforensics/FF++
```

### 4) Train - 4 Modes (Multi GPU)

```bash
torchrun --nproc_per_node=4 scripts/train_epoch_based.py --dataset-path ./data/faceforensics/FF++
torchrun --nproc_per_node=4 scripts/train_convergence_normal.py --dataset-path ./data/faceforensics/FF++
torchrun --nproc_per_node=4 scripts/train_hashring_epoch.py --dataset-path ./data/faceforensics/FF++
torchrun --nproc_per_node=4 scripts/train_convergence_hashring.py --dataset-path ./data/faceforensics/FF++
```

### 5) Resume Training

```bash
python scripts/train_convergence_hashring.py \
	--dataset-path ./data/faceforensics/FF++ \
	--checkpoint-dir ./checkpoints/convergence_hashring \
	--resume
```

### 6) Mode-Specific Shell Runners

```bash
bash scripts/run_faceforensics_epoch.sh
bash scripts/run_faceforensics_convergence.sh
bash scripts/run_faceforensics_hashring_epoch.sh
bash scripts/run_faceforensics_convergence_hashring.sh
```

### 7) Reproducible Fault-Tolerance Metrics Runs

Use the shell runners with an experiment profile.

#### Baseline Runs

These runs measure no-failure training cost and model quality:

```bash
EXPERIMENT_PROFILE=baseline bash scripts/run_faceforensics_epoch.sh
EXPERIMENT_PROFILE=baseline bash scripts/run_faceforensics_convergence.sh
EXPERIMENT_PROFILE=baseline bash scripts/run_faceforensics_hashring_epoch.sh
EXPERIMENT_PROFILE=baseline bash scripts/run_faceforensics_convergence_hashring.sh
```

#### Resilience Runs

These runs inject a runtime fault, restart automatically, and capture recovery metrics:

```bash
EXPERIMENT_PROFILE=resilience bash scripts/run_faceforensics_epoch.sh
EXPERIMENT_PROFILE=resilience bash scripts/run_faceforensics_convergence.sh
EXPERIMENT_PROFILE=resilience bash scripts/run_faceforensics_hashring_epoch.sh
EXPERIMENT_PROFILE=resilience bash scripts/run_faceforensics_convergence_hashring.sh
```

#### Failure Timing Sweep

Early fault:

```bash
EXPERIMENT_PROFILE=resilience RUNTIME_FAULT_AFTER_CHECKPOINTS=1 bash scripts/run_faceforensics_epoch.sh
EXPERIMENT_PROFILE=resilience RUNTIME_FAULT_AFTER_CHECKPOINTS=1 bash scripts/run_faceforensics_convergence.sh
EXPERIMENT_PROFILE=resilience RUNTIME_FAULT_AFTER_CHECKPOINTS=1 bash scripts/run_faceforensics_hashring_epoch.sh
EXPERIMENT_PROFILE=resilience RUNTIME_FAULT_AFTER_CHECKPOINTS=1 bash scripts/run_faceforensics_convergence_hashring.sh
```

Mid fault:

```bash
EXPERIMENT_PROFILE=resilience RUNTIME_FAULT_AFTER_CHECKPOINTS=3 bash scripts/run_faceforensics_epoch.sh
EXPERIMENT_PROFILE=resilience RUNTIME_FAULT_AFTER_CHECKPOINTS=3 bash scripts/run_faceforensics_convergence.sh
EXPERIMENT_PROFILE=resilience RUNTIME_FAULT_AFTER_CHECKPOINTS=3 bash scripts/run_faceforensics_hashring_epoch.sh
EXPERIMENT_PROFILE=resilience RUNTIME_FAULT_AFTER_CHECKPOINTS=3 bash scripts/run_faceforensics_convergence_hashring.sh
```

Late fault:

```bash
EXPERIMENT_PROFILE=resilience RUNTIME_FAULT_AFTER_CHECKPOINTS=5 bash scripts/run_faceforensics_epoch.sh
EXPERIMENT_PROFILE=resilience RUNTIME_FAULT_AFTER_CHECKPOINTS=5 bash scripts/run_faceforensics_convergence.sh
EXPERIMENT_PROFILE=resilience RUNTIME_FAULT_AFTER_CHECKPOINTS=5 bash scripts/run_faceforensics_hashring_epoch.sh
EXPERIMENT_PROFILE=resilience RUNTIME_FAULT_AFTER_CHECKPOINTS=5 bash scripts/run_faceforensics_convergence_hashring.sh
```

Run each resilience configuration at least 3 times, ideally 5 times, before comparing methods.

#### Metrics Output

Use these outputs for analysis:

- [experiment_results/experiment_summaries.jsonl](experiment_results/experiment_summaries.jsonl)
- per-run JSON, Markdown, HTML, and CSV files under `experiment_results/`
- [RESILIENCE_EXPERIMENT_PROTOCOL.md](RESILIENCE_EXPERIMENT_PROTOCOL.md)

Important resilience metrics to compare:

- `resume_succeeded`
- `time_to_resume_sec`
- `rollback_epochs`
- `rollback_steps`
- `resume_load_source`
- `local_cache_loads`
- `central_storage_loads`
- `recovered_successfully`
- `checkpoint_overhead_percent`
- `final_val_accuracy`

### 8) Fault Injection Test

```bash
bash scripts/test_fault_tolerance.sh 1
bash scripts/test_fault_tolerance.sh 2
```

### 9) Common Useful Flags

```bash
--epochs 20 --batch-size 2 --num-workers 4
--checkpoint-dir ./checkpoints/exp1 --checkpoint-interval 5
--failure-rate-lambda 0.00208 --checkpoint-cost-sec 0.60
--fit-interval-steps 100 --min-convergence-interval-sec 5 --max-convergence-interval-sec 1800
--hash-ring-virtual-nodes 100 --hash-ring-cache-dir ./checkpoints/hash_cache
--fast-video-mode --num-frames 16
```

## What This Project Implements

This codebase supports four checkpointing modes:

1. Pure epoch based
2. Pure convergence aware + normal storage
3. Hash ring + epoch based
4. Convergence timing + hash ring storage/recovery

The implementation is layered so existing epoch-based behavior remains available.

## Core Concepts

### 1) Convergence-Aware Checkpointing (When)

Instead of fixed time/epoch spacing, checkpoints can be scheduled based on observed loss dynamics.

- Online fitting tracks loss trend
- Checkpoint interval adapts over training
- Dense checkpoints early, sparser checkpoints later

Primary implementation:

- [src/coci/checkpointing/convergence_scheduler.py](src/coci/checkpointing/convergence_scheduler.py)

Methodology reference in this repo:

- [CONVERGENCE_AWARE_CHECKPOINTING.md](CONVERGENCE_AWARE_CHECKPOINTING.md)

### 2) Hash Ring Checkpointing (Where/How)

Checkpoints are mapped to nodes through a hash ring and cached locally for faster recovery.

- Consistent hashing with virtual nodes
- Shard ownership and local cache management
- Resilient fallback to central checkpoint storage

Primary implementation:

- [src/coci/checkpointing/hash_ring_checkpoint_manager.py](src/coci/checkpointing/hash_ring_checkpoint_manager.py)
- [src/coci/hashing/hash_ring.py](src/coci/hashing/hash_ring.py)
- [src/coci/hashing/recovery_scheduler.py](src/coci/hashing/recovery_scheduler.py)

Methodology reference in this repo:

- [HASH_RING.md](HASH_RING.md)

## Main Training Entry Point

Primary trainer:

- [scripts/train_faceforensics.py](scripts/train_faceforensics.py)

This script supports:

- single GPU and multi-GPU (torchrun)
- image mode and video mode
- fault injection
- checkpoint resume
- all 4 checkpointing modes

## Mode Matrix

| Mode | Timing (When) | Placement/Recovery (Where/How) | Launcher |
|---|---|---|---|
| `epoch` | Epoch interval | Normal checkpoint storage | [scripts/train_epoch_based.py](scripts/train_epoch_based.py) |
| `convergence` | Convergence-aware scheduler | Normal checkpoint storage | [scripts/train_convergence_normal.py](scripts/train_convergence_normal.py) |
| `hash-ring-epoch` | Epoch interval | Hash ring + local shard cache | [scripts/train_hashring_epoch.py](scripts/train_hashring_epoch.py) |
| `convergence-hash-ring` | Convergence-aware scheduler | Hash ring + local shard cache | [scripts/train_convergence_hashring.py](scripts/train_convergence_hashring.py) |

## Project Layout

```text
adaptive-ring-checkpointing/
	scripts/
		train_faceforensics.py
		train_epoch_based.py
		train_convergence_normal.py
		train_hashring_epoch.py
		train_convergence_hashring.py
		run_faceforensics.sh
		run_faceforensics_epoch.sh
		run_faceforensics_convergence.sh
		run_faceforensics_hashring_epoch.sh
		run_faceforensics_convergence_hashring.sh
		test_fault_tolerance.sh
	src/coci/
		checkpointing/
			checkpoint_manager.py
			hash_ring_checkpoint_manager.py
			convergence_scheduler.py
			strategy.py
		hashing/
			hash_ring.py
			fault_detector.py
			recovery_scheduler.py
		data_ingestor/
			faceforensics.py
			cifar.py
		distributed.py
	configs/
		dev.yaml
		server.yaml
```

## Requirements

- Python 3.10+
- PyTorch + torchvision
- CUDA-compatible GPU for multi-GPU training
- FaceForensics++ dataset (or use download flow)

Project metadata and dependency sources:

- [pyproject.toml](pyproject.toml)
- [requirements.txt](requirements.txt)

## Installation

### Option A: pip + venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option B: uv (if you use uv)

```bash
uv sync
```

## Dataset Setup

Use built-in dataset download helper:

```bash
python scripts/train_faceforensics.py --download-dataset
```

Or point to an existing dataset path:

```bash
python scripts/train_faceforensics.py --dataset-path ./data/faceforensics/FF++
```

## Quick Start

### A) Run mode-specific Python launchers

1. Pure epoch based

```bash
python scripts/train_epoch_based.py --dataset-path ./data/faceforensics/FF++
```

2. Pure convergence aware + normal storage

```bash
python scripts/train_convergence_normal.py --dataset-path ./data/faceforensics/FF++
```

3. Hash ring + epoch based

```bash
python scripts/train_hashring_epoch.py --dataset-path ./data/faceforensics/FF++
```

4. Convergence + hash ring

```bash
python scripts/train_convergence_hashring.py --dataset-path ./data/faceforensics/FF++
```

### B) Run mode-specific shell runners

These scripts include GPU detection, environment activation, and restart loops:

- [scripts/run_faceforensics_epoch.sh](scripts/run_faceforensics_epoch.sh)
- [scripts/run_faceforensics_convergence.sh](scripts/run_faceforensics_convergence.sh)
- [scripts/run_faceforensics_hashring_epoch.sh](scripts/run_faceforensics_hashring_epoch.sh)
- [scripts/run_faceforensics_convergence_hashring.sh](scripts/run_faceforensics_convergence_hashring.sh)

Examples:

```bash
bash scripts/run_faceforensics_epoch.sh
bash scripts/run_faceforensics_convergence.sh
bash scripts/run_faceforensics_hashring_epoch.sh
bash scripts/run_faceforensics_convergence_hashring.sh
```

## Important Training Arguments

Common:

- `--dataset-path`
- `--epochs`
- `--batch-size`
- `--num-workers`
- `--checkpoint-dir`
- `--checkpoint-interval`
- `--resume`

Mode selection:

- `--training-mode`
	- `epoch`
	- `convergence`
	- `hash-ring-epoch`
	- `convergence-hash-ring`

Convergence-aware controls:

- `--failure-rate-lambda`
- `--checkpoint-cost-sec`
- `--fit-interval-steps`
- `--min-convergence-interval-sec`
- `--max-convergence-interval-sec`

Hash ring controls:

- `--hash-ring-virtual-nodes`
- `--hash-ring-cache-dir`

Video mode controls:

- `--video-mode`
- `--fast-video-mode`
- `--num-frames`
- `--temporal-model`
- `--extract-frames`

Fault injection:

- `--inject-fault`
- `--inject-rank`
- `--inject-rate`

## Distributed Training

Use torchrun for multi-GPU:

```bash
torchrun --nproc_per_node=4 scripts/train_convergence_hashring.py \
	--dataset-path ./data/faceforensics/FF++ \
	--epochs 20 \
	--batch-size 2
```

## Checkpointing Behavior Notes

- Base manager supports both legacy epoch checkpoints and custom checkpoint IDs:
	- [src/coci/checkpointing/checkpoint_manager.py](src/coci/checkpointing/checkpoint_manager.py)
- Hash ring manager layers caching over canonical checkpoints:
	- [src/coci/checkpointing/hash_ring_checkpoint_manager.py](src/coci/checkpointing/hash_ring_checkpoint_manager.py)
- Resume loads latest checkpoint across `checkpoint_*.pt` artifacts.

## Fault Tolerance Verification

End-to-end validation docs:

- [VERIFICATION_E2E.md](VERIFICATION_E2E.md)
- [VERIFICATION_RESULTS.md](VERIFICATION_RESULTS.md)

Test script:

- [scripts/test_fault_tolerance.sh](scripts/test_fault_tolerance.sh)

Run:

```bash
bash scripts/test_fault_tolerance.sh 1
bash scripts/test_fault_tolerance.sh 2
```

## Logs and Experiment Artifacts

Expected artifacts include:

- checkpoints under configured checkpoint directory
- JSONL logs in project root (strategy/mode dependent)
- graph outputs under `graphify-out/` when graphify is run

## Known Environment Notes

- Shell runners (`*.sh`) are for Linux/Unix environments.
- On Windows, prefer direct Python launchers or run through WSL/Git Bash.
- Multi-GPU runs require a working NCCL/CUDA setup.

## Troubleshooting

1. Import errors from script launchers

- Use launchers in [scripts](scripts) exactly as provided.
- Ensure project root is current working directory.

2. No checkpoint found on resume

- Confirm `--checkpoint-dir` matches prior run.
- Verify files named `checkpoint_*.pt` exist.

3. Multi-GPU hangs

- Verify all ranks can reach each other.
- Check NCCL environment and GPU visibility.
- Start with `--nproc_per_node=1` and scale up.

4. Dataset path errors

- Run `--download-dataset` first, or provide valid `--dataset-path`.

## Related Documents

- [HASH_RING.md](HASH_RING.md)
- [CONVERGENCE_AWARE_CHECKPOINTING.md](CONVERGENCE_AWARE_CHECKPOINTING.md)
- [VERIFICATION_E2E.md](VERIFICATION_E2E.md)
- [VERIFICATION_RESULTS.md](VERIFICATION_RESULTS.md)

## License

No license file is currently present in this repository.

