# Experiment Results Organization

## Overview

This directory contains organized experiment results from training runs for the Adaptive Ring Checkpointing project. Experiments are organized using a flat naming convention that makes it easy to identify the training mode, framework, and timestamp at a glance.

The organization structure was established to replace the original timestamp-based naming (`faceforensics_YYYYMMDD_HHMMSS`) with a more descriptive convention that encodes the training mode and framework directly in the directory name.

## Directory Naming Convention

All experiment directories follow this format:

```
{mode}_{framework}_{date}_{time}
```

### Components

| Component | Description | Examples |
|-----------|-------------|---------|
| `mode` | Training mode or strategy | `epoch`, `convergence`, `epoch_hashring`, `convergence_hashring` |
| `framework` | Training framework used | `vanilla`, `deepspeed`, `fsdp`, `lightning`, `hf_trainer`, `wandb` |
| `date` | Date of experiment (YYYYMMDD) | `20260419`, `20260430` |
| `time` | Time of experiment (HHMMSS) | `101309`, `175303` |

### Valid Modes

- `epoch` - Standard epoch-based training
- `convergence` - Convergence-based checkpointing (fixed step interval)
- `epoch_hashring` - Epoch-based training with hash ring fault tolerance
- `convergence_hashring` - Convergence-based with hash ring fault tolerance

### Valid Frameworks

- `vanilla` - Standard PyTorch training loop
- `deepspeed` - DeepSpeed distributed training
- `fsdp` - PyTorch FSDP (Fully Sharded Data Parallel)
- `lightning` - PyTorch Lightning
- `hf_trainer` - HuggingFace Trainer
- `wandb` - Weights & Biases artifacts integration

## Examples

Here are some example directory names and what they represent:

| Directory Name | Description |
|---------------|-------------|
| `epoch_vanilla_20260419_101309` | Epoch-based training, vanilla PyTorch, run on April 19, 2026 at 10:13:09 |
| `convergence_vanilla_20260419_104954` | Convergence-based checkpointing, vanilla PyTorch |
| `epoch_hashring_vanilla_20260419_112016` | Epoch-based with hash ring fault tolerance |
| `convergence_hashring_vanilla_20260419_113143` | Convergence-based with hash ring |
| `epoch_deepspeed_20260430_175303` | Epoch-based training using DeepSpeed framework |
| `epoch_fsdp_20260430_182642` | Epoch-based training using PyTorch FSDP |
| `epoch_lightning_20260430_172607` | Epoch-based training using PyTorch Lightning |
| `epoch_hf_trainer_20260430_174300` | Epoch-based training using HuggingFace Trainer |
| `epoch_wandb_20260430_184333` | Epoch-based training with W&B artifacts |

## Directory Contents

Each experiment directory typically contains:

| File Pattern | Description |
|-------------|-------------|
| `epoch_*.html` | Training report with metrics visualization |
| `epoch_*.json` | Metrics data in JSON format |
| `epoch_*.md` | Summary of the training run |
| `epoch_*.csv` | CSV export of training metrics |

## Index Files

The `index/` subdirectory contains metadata and migration tools:

| File | Purpose |
|------|---------|
| `experiment_register.json` | Maps old experiment names to new organized names with parsed metadata |
| `migration_script.py` | Tool to migrate experiments from old naming to new convention |
| `experiment_helpers.py` | Helper functions for generating consistent experiment names |

## How to Find Experiments

### By Training Mode

Look for directories starting with the mode name:
```bash
ls -d experiment_results/epoch_*
ls -d experiment_results/convergence_*
ls -d experiment_results/*_hashring_*
```

### By Framework

Look for directories containing the framework name:
```bash
ls -d experiment_results/*_vanilla_*
ls -d experiment_results/*_deepspeed_*
ls -d experiment_results/*_fsdp_*
```

### By Date

All directories include the date in YYYYMMDD format:
```bash
ls -d experiment_results/*_20260419_*
ls -d experiment_results/*_20260430_*
```

## Adding New Experiments

### Option 1: Use the Helper Script

The `experiment_helpers.py` script provides functions to generate consistent experiment names:

```bash
python experiment_results/index/experiment_helpers.py generate --mode epoch --framework vanilla
```

This will output a name like: `epoch_vanilla_20260501_162107`

### Option 2: Follow the Convention Manually

When creating a new experiment directory, follow this pattern:
```
{mode}_{framework}_{YYYYMMDD}_{HHMMSS}
```

Example:
```bash
mkdir experiment_results/epoch_vanilla_$(date +%Y%m%d_%H%M%S)
```

### Integrating with Training Scripts

The training scripts in `scripts/` can be updated to use the helper function for consistent naming:

```python
from experiment_results.index.experiment_helpers import generate_experiment_name
import time

# Generate experiment name
exp_name = generate_experiment_name(
    mode="epoch",
    framework="vanilla",
    timestamp=datetime.now()
)

# Create experiment directory
os.makedirs(f"experiment_results/{exp_name}", exist_ok=True)
```

## Migration from Old Naming

If you have experiments using the old naming convention (`faceforensics_YYYYMMDD_HHMMSS`), use the migration script:

```bash
# Preview changes (dry run)
python experiment_results/index/migration_script.py --dry-run

# Execute the migration
python experiment_results/index/migration_script.py --execute
```

The migration will:
1. Parse `experiment_summaries.jsonl` to extract metadata
2. Generate new names following the convention
3. Rename directories
4. Update `experiment_summaries.jsonl` with new paths
5. Create a backup before making changes

## Configuration

The naming convention is configurable via `configs/experiment_naming.yaml`. You can modify valid modes, frameworks, and date/time formats there.

## Quick Reference

| Need to... | Do this... |
|------------|------------|
| Find all epoch-based experiments | `ls -d experiment_results/epoch_*` |
| Find all DeepSpeed experiments | `ls -d experiment_results/*_deepspeed_*` |
| Find experiments from a specific date | `ls -d experiment_results/*_20260419_*` |
| Generate a new experiment name | `python experiment_results/index/experiment_helpers.py generate --mode epoch --framework vanilla` |
| See all experiments as JSON | `cat experiment_results/index/experiment_register.json` |
| Migrate old experiments | `python experiment_results/index/migration_script.py --execute` |
