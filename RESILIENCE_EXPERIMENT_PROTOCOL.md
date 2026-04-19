# Resilience Experiment Protocol

This protocol is for proving that `convergence-hash-ring` is a better full-fledged checkpoint mechanism than plain epoch checkpointing.

## Goal

Demonstrate that the combined method improves failure handling, not just no-failure training.

The claim to validate is:

`convergence-aware checkpoint timing + hash-ring recovery reduces lost work and restart cost under failures while preserving final model quality.`

## What To Compare

Run the same workload in all four modes:

1. `epoch`
2. `convergence`
3. `hash-ring-epoch`
4. `convergence-hash-ring`

Use identical:

- dataset split
- model
- optimizer
- learning rate
- batch size
- epoch count
- failure trigger schedule

## Experiment Families

### 1. No-Failure Baseline

Purpose:

- measure steady-state checkpoint overhead
- confirm similar model quality

Report:

- `total_training_time_sec`
- `checkpoint_overhead_percent`
- `total_checkpoint_count`
- `best_val_accuracy`
- `final_val_accuracy`

### 2. Single Runtime Fault

Run each mode with a crash after a checkpoint is saved:

```bash
python scripts/train_faceforensics.py \
  --training-mode epoch \
  --runtime-fault-injection \
  --runtime-fault-after-checkpoints 1 \
  --checkpoint-dir ./checkpoints/resilience_epoch
```

Then resume:

```bash
python scripts/train_faceforensics.py \
  --training-mode epoch \
  --resume \
  --checkpoint-dir ./checkpoints/resilience_epoch
```

Purpose:

- measure time to resume
- measure rollback distance
- validate successful recovery

Report:

- `resume_attempted`
- `resume_succeeded`
- `time_to_resume_sec`
- `rollback_epochs`
- `rollback_steps`
- `runtime_fault_checkpoint_id`
- `resumed_checkpoint_id`

### 3. Failure Timing Sweep

Repeat the runtime-fault experiment with failures placed:

- early
- mid-training
- late

Suggested checkpoints:

- after first saved checkpoint
- after third saved checkpoint
- after fifth saved checkpoint

Purpose:

- show convergence-aware timing reduces rollback sensitivity to failure timing

### 4. Hash-Ring Recovery Activation

Use hash-ring modes and resume from the same checkpoint directory:

```bash
python scripts/train_faceforensics.py \
  --training-mode convergence-hash-ring \
  --runtime-fault-injection \
  --runtime-fault-after-checkpoints 1 \
  --checkpoint-dir ./checkpoints/resilience_conv_hash \
  --hash-ring-cache-dir ./checkpoints/resilience_conv_hash_cache
```

Resume:

```bash
python scripts/train_faceforensics.py \
  --training-mode convergence-hash-ring \
  --resume \
  --checkpoint-dir ./checkpoints/resilience_conv_hash \
  --hash-ring-cache-dir ./checkpoints/resilience_conv_hash_cache
```

Purpose:

- distinguish local-cache resume from central-storage fallback
- quantify the hash-ring recovery path

Report:

- `resume_load_source`
- `last_load_source`
- `local_cache_loads`
- `central_storage_loads`
- `cache_write_count`
- `cache_hit_rate_percent`

## Recommended Trial Count

For each scenario, run at least:

- `3` trials for quick evidence
- `5` trials for stronger comparison

Aggregate results from:

- [experiment_results/experiment_summaries.jsonl](/D:/Sai/HPC/project/adaptive-ring-checkpointing/experiment_results/experiment_summaries.jsonl)

## Decision Rules

You have evidence that the combined method is better if it shows:

- similar final accuracy to `epoch`
- lower rollback than `epoch`
- lower or comparable `time_to_resume_sec`
- more resumes from `local-cache` than central storage
- fewer central storage loads after failure

## Metrics To Highlight In Plots

Make these plots first:

1. `checkpoint_overhead_percent` by mode
2. `time_to_resume_sec` by mode
3. `rollback_epochs` by failure timing and mode
4. `central_storage_loads` vs `local_cache_loads` for hash-ring modes
5. `final_val_accuracy` by mode

## Reporting Template

Use a table like this in your results section:

| Metric | epoch | convergence | hash-ring-epoch | convergence-hash-ring |
|---|---:|---:|---:|---:|
| Final Val Accuracy (%) |  |  |  |  |
| Checkpoint Overhead (%) |  |  |  |  |
| Time to Resume (s) |  |  |  |  |
| Rollback Epochs |  |  |  |  |
| Rollback Steps |  |  |  |  |
| Local Cache Loads |  |  |  |  |
| Central Storage Loads |  |  |  |  |
| Recovery Success Rate (%) |  |  |  |  |

## Interpretation Guidance

If `convergence-hash-ring` has higher no-failure overhead but better recovery metrics, that is still a valid resilience win.

The core argument is not:

`it trains faster in the happy path`

The core argument is:

`it loses less work and resumes more efficiently when failures actually happen`
