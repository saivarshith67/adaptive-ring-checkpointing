# Metrics Collection and Export Guide

## Overview

The project now includes a comprehensive metrics collection and export system that tracks performance across all four training modes:

1. **Epoch-based checkpointing** - Fixed checkpoint frequency per epoch
2. **Convergence-aware (COCI)** - Adaptive checkpoint timing based on convergence
3. **Hash-ring-epoch** - Ring-based shard distribution with epoch checkpoints
4. **Convergence-hash-ring** - COCI + Hash-ring combined approach

## Available Metrics

### Training Performance Metrics
- **train_loss** / **val_loss**: Per-epoch training and validation loss
- **train_accuracy** / **val_accuracy**: Per-epoch accuracy
- **best_val_accuracy**: Highest validation accuracy achieved
- **final_val_accuracy**: Final validation accuracy
- **convergence_speed**: How quickly loss decreases

### Checkpointing Metrics
- **total_checkpoint_count**: Total number of checkpoints saved
- **total_checkpoint_time_sec**: Cumulative checkpoint save time
- **checkpoint_overhead_percent**: % of training time spent checkpointing
- **avg_checkpoint_size_mb**: Average checkpoint size
- **checkpoint_frequency**: Checkpoints per epoch (or time interval)

### Convergence Scheduler Metrics (COCI modes only)
- **current_interval_sec**: Current checkpoint interval in seconds
- **theta1**: Exponential decay parameter (loss model)
- **theta2**: Exponential offset parameter (loss model)
- **loss_fit_error**: Quality of exponential fit
- **total_convergence_checkpoints**: Number of adaptive checkpoints

### Hash Ring Metrics (Hash-ring modes only)
- **cache_hits**: Number of local cache hits
- **cache_misses**: Number of local cache misses
- **cache_hit_rate_percent**: Hit rate percentage
- **cache_size_mb**: Current cache size
- **total_shards**: Total shards in the ring
- **shards_owned**: Shards owned by this node
- **shard_recovery_count**: How many shards were recovered
- **recovery_time_sec**: Time spent on recovery

### Fault Tolerance Metrics
- **injected_faults**: Number of faults intentionally injected
- **detected_faults**: Number of faults detected
- **recovered_successfully**: Successful recovery count
- **recovery_failures**: Failed recovery count
- **total_recovery_time_sec**: Total time spent in recovery

### Distributed Training Metrics
- **world_size**: Number of GPUs/ranks
- **num_syncs**: Number of collective synchronization operations
- **total_sync_time_sec**: Total time spent in synchronization
- **comm_overhead_percent**: % of training time spent in communication

## Export Formats

The system supports multiple export formats:

### 1. JSON (Full Per-Run Details)
- **Purpose**: Complete details for a single training run
- **Location**: `experiment_results/{experiment_name}/{mode}_YYYYMMDD_HHMMSS.json`
- **Contents**: Summary metrics + per-epoch metrics
- **Use case**: Detailed analysis of individual runs

**Example:**
```json
{
  "timestamp": "2026-04-18T10:30:00",
  "summary": {
    "experiment_name": "faceforensics_exp1",
    "training_mode": "convergence-hash-ring",
    "best_val_accuracy": 92.5,
    "checkpoint_overhead_percent": 8.2,
    ...
  },
  "epoch_metrics": [
    {
      "epoch": 0,
      "train_loss": 0.5234,
      "train_accuracy": 75.3,
      "val_loss": 0.4823,
      "val_accuracy": 78.1,
      ...
    },
    ...
  ]
}
```

### 2. JSONL (Aggregated Results)
- **Purpose**: Append-only format for aggregating multiple runs
- **Location**: `experiment_results/experiment_summaries.jsonl`
- **Contents**: One-line summary per run (flattened for easy analysis)
- **Use case**: Statistical comparison, plotting trends

**Example:**
```
{"timestamp": "2026-04-18T10:30:00", "experiment_name": "faceforensics_exp1", "training_mode": "epoch", "total_training_time_sec": 1234.5, "checkpoint_overhead_percent": 5.2, ...}
{"timestamp": "2026-04-18T10:45:00", "experiment_name": "faceforensics_exp1", "training_mode": "convergence", "total_training_time_sec": 1189.3, "checkpoint_overhead_percent": 4.8, ...}
```

### 3. CSV (Per-Epoch Metrics)
- **Purpose**: Easy import into plotting tools
- **Location**: `experiment_results/{experiment_name}/{mode}_epochs_YYYYMMDD_HHMMSS.csv`
- **Contents**: One row per epoch with all per-epoch metrics
- **Use case**: Plotting loss/accuracy curves

**Example:**
```
epoch,train_loss,train_accuracy,val_loss,val_accuracy,epoch_duration_sec,checkpoint_count
0,0.5234,75.3,0.4823,78.1,45.2,1
1,0.4123,82.1,0.3891,85.2,42.8,1
```

### 4. HTML Report
- **Purpose**: Visual summary with formatted tables
- **Location**: `experiment_results/{experiment_name}/{mode}_YYYYMMDD_HHMMSS.html`
- **Contents**: All metrics in styled HTML format
- **Use case**: Quick review, sharing results

### 5. Markdown Report
- **Purpose**: Version-control friendly, readable format
- **Location**: `experiment_results/{experiment_name}/{mode}_YYYYMMDD_HHMMSS.md`
- **Contents**: All metrics in Markdown tables
- **Use case**: Documentation, Git tracking

## Integration into Training Scripts

### Step 1: Initialize Collector
```python
from src.coci.metrics import MetricsCollector, MetricsExporter

# Create collector
metrics = MetricsCollector(
    experiment_name="faceforensics_exp1",
    training_mode=args.training_mode,
    dataset="faceforensics",
    model="efficientnet-b0",
    epochs=args.epochs,
    batch_size=args.batch_size
)

# Set distributed training info
if is_distributed_initialized():
    metrics.set_distributed_config(get_world_size())
```

### Step 2: Record Per-Epoch Metrics
```python
# In training loop, at end of each epoch:
metrics.start_epoch(epoch)

# ... training code ...

metrics.end_epoch(
    epoch=epoch,
    train_loss=avg_train_loss,
    train_acc=avg_train_acc,
    val_loss=avg_val_loss,
    val_acc=avg_val_acc
)
```

### Step 3: Record Checkpoint Events
```python
# When saving checkpoint:
import os
checkpoint_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)

metrics.record_checkpoint(
    checkpoint_size_mb=checkpoint_size_mb,
    save_time_sec=save_time,
    checkpoint_id=f"step_{global_step:012d}"
)
```

### Step 4: Record Convergence Scheduler Events (if using COCI)
```python
# When convergence scheduler updates:
metrics.update_convergence_metrics(
    current_interval_sec=convergence_scheduler.current_interval_seconds,
    theta1=fit.theta1,
    theta2=fit.theta2,
    loss_fit_error=compute_fit_error(fit)
)

# When checkpoint is triggered by convergence:
metrics.increment_convergence_checkpoint()
```

### Step 5: Record Hash Ring Events (if using hash ring)
```python
# Update hash ring stats:
metrics.update_hash_ring_metrics(
    total_shards=shard_manager.num_shards(),
    cache_size_mb=shard_manager.cache_size_bytes() / (1024 * 1024),
    max_cache_mb=args.max_cache_mb,
    shards_owned=shard_manager.count_owned_shards()
)

# Track cache operations:
if cache_hit:
    metrics.increment_cache_hit()
else:
    metrics.increment_cache_miss()
```

### Step 6: Record Fault Events (if using fault injection)
```python
# When fault is injected:
metrics.increment_injected_fault()

# When fault is detected:
metrics.increment_detected_fault()

# When recovery happens:
metrics.record_recovery(
    success=recovery_succeeded,
    recovery_time_sec=recovery_time
)
```

### Step 7: Export Metrics
```python
# At end of training:
exporter = MetricsExporter(base_export_dir="./experiment_results")

# Export in multiple formats
summary_json_path = exporter.export_summary_json(metrics)
summary_jsonl_path = exporter.export_summary_jsonl(metrics)
epochs_csv_path = exporter.export_epoch_metrics_csv(metrics)
html_path = exporter.export_html_report(metrics)
md_path = exporter.export_markdown_report(metrics)

print(f"Results exported to {summary_json_path}")
```

## Comparing Methods

### Load and Compare Results
```python
from src.coci.metrics import MetricsExporter

exporter = MetricsExporter()

# Load all results from JSONL
results = exporter.load_jsonl_results("./experiment_results/experiment_summaries.jsonl")

# Get comparison statistics by mode
comparison = exporter.compare_modes(results)

# Print comparison
for mode, stats in comparison.items():
    print(f"\n{mode}:")
    print(f"  Runs: {stats['num_runs']}")
    print(f"  Avg Training Time: {stats['avg_training_time_sec']:.1f}s")
    print(f"  Avg Best Val Acc: {stats['avg_best_val_accuracy']:.2f}%")
    print(f"  Avg Checkpoint Overhead: {stats['avg_checkpoint_overhead_percent']:.1f}%")
```

### Key Metrics to Compare

1. **Checkpoint Overhead (primary)**: Lower is better
   - `checkpoint_overhead_percent` (% of training time)
   - Ratio: `total_checkpoint_time_sec / total_training_time_sec`

2. **Training Performance**: Higher is better
   - `best_val_accuracy` (accuracy percentage)
   - `convergence_speed` (epochs to reach target loss)

3. **Convergence Efficiency** (for COCI modes):
   - `total_convergence_checkpoints` vs `total_checkpoint_count`
   - `current_interval_sec` (should adapt based on loss)

4. **Hash Ring Efficiency** (for hash-ring modes):
   - `cache_hit_rate_percent` (higher = better locality)
   - `recovery_time_sec` vs `total_recovery_time_sec`

5. **Fault Tolerance**:
   - `recovery_failures / (recovered_successfully + recovery_failures)`
   - `total_recovery_time_sec` (time overhead)

## Plotting Examples

### Using CSV for Matplotlib
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load epoch metrics
df = pd.read_csv("experiment_results/faceforensics_exp1/convergence-hash-ring_epochs_*.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss curve
axes[0].plot(df['epoch'], df['train_loss'], label='Train', marker='o')
axes[0].plot(df['epoch'], df['val_loss'], label='Val', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Convergence')
axes[0].legend()
axes[0].grid(True)

# Accuracy curve
axes[1].plot(df['epoch'], df['train_accuracy'], label='Train', marker='o')
axes[1].plot(df['epoch'], df['val_accuracy'], label='Val', marker='s')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy Convergence')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('convergence_curves.png')
```

### Using JSONL for Comparison
```python
import pandas as pd
import matplotlib.pyplot as plt
from src.coci.metrics import MetricsExporter

exporter = MetricsExporter()
results = exporter.load_jsonl_results("./experiment_results/experiment_summaries.jsonl")
df = pd.DataFrame(results)

# Filter by experiment
df_exp = df[df['experiment_name'] == 'faceforensics_exp1']

# Group by mode
grouped = df_exp.groupby('training_mode')[['checkpoint_overhead_percent', 'best_val_accuracy']].mean()

fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(grouped))
ax.bar([i - 0.2 for i in x], grouped['checkpoint_overhead_percent'], width=0.4, label='Checkpoint Overhead %')
ax.bar([i + 0.2 for i in x], grouped['best_val_accuracy']/100, width=0.4, label='Best Val Acc (normalized)')
ax.set_xticks(x)
ax.set_xticklabels(grouped.index)
ax.set_ylabel('Value')
ax.set_title('Method Comparison')
ax.legend()
plt.tight_layout()
plt.savefig('method_comparison.png')
```

## Directory Structure

After running experiments with metrics enabled:

```
experiment_results/
├── faceforensics_exp1/
│   ├── epoch_20260418_103000.json
│   ├── epoch_20260418_103000_epochs.csv
│   ├── epoch_20260418_103000.html
│   ├── epoch_20260418_103000.md
│   ├── convergence_20260418_110000.json
│   ├── convergence_20260418_110000_epochs.csv
│   ├── hash-ring-epoch_20260418_112000.json
│   └── convergence-hash-ring_20260418_114000.json
└── experiment_summaries.jsonl  (appended to across runs)
```

## Best Practices

1. **Use consistent experiment names**: Helps organize results
2. **Run multiple seeds**: Average results in comparison for robustness
3. **Export to JSONL first**: Append-only format is safer than overwriting
4. **Review HTML reports**: Quick visual sanity check of results
5. **Use CSV for plotting**: Easiest to work with plotting libraries
6. **Archive results**: Copy experiment_results/ to archive after important runs
7. **Version control markdown reports**: Track results evolution with Git

## Example Script

See `scripts/run_with_metrics.py` for a complete example showing:
- How to initialize metrics
- Run all four training modes
- Export results in all formats
- Generate comparison plots
- Save final analysis

## Troubleshooting

**Q: Metrics not being recorded?**
- Ensure `MetricsCollector` is initialized before training
- Check that `record_*` methods are called at appropriate points
- Verify collector is passed to all relevant components

**Q: Export directory not created?**
- MetricsExporter automatically creates directories
- Check write permissions to `experiment_results/`

**Q: Cache hit rate missing for non-hash-ring modes?**
- Hash ring metrics are only populated for hash-ring modes
- Compare only relevant metrics for each mode

**Q: JSONL file growing too large?**
- JSONL is append-only by design
- Consider archiving old results periodically
- Or analyze and reset with new experiment name
