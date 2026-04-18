# Recommended Comparison Metrics

## Summary Table of Key Metrics

This document provides the **recommended metrics to compare** the four training methods.

## The Four Methods

| Mode | Description | Key Differentiator |
|------|-------------|--------------------|
| **epoch** | Fixed checkpoint every N epochs | Baseline: simple, predictable |
| **convergence** | Adaptive checkpoint based on convergence (COCI) | Convergence-aware timing |
| **hash-ring-epoch** | Hash-ring shard distribution + epoch checkpoints | Ring-based locality |
| **convergence-hash-ring** | COCI + Hash-ring combined | Best of both worlds |

## Primary Comparison Metrics

### 1. **Checkpoint Efficiency** (Most Important)
Measure how much training time is wasted on checkpointing.

| Metric | Unit | Direction | Capture From |
|--------|------|-----------|--------------|
| `checkpoint_overhead_percent` | % | Lower is better | summary.checkpoint_overhead_percent |
| `total_checkpoint_time_sec` | seconds | Lower is better | summary.total_checkpoint_time_sec |
| `total_checkpoint_count` | count | Fewer better (more selective) | summary.total_checkpoint_count |
| `avg_checkpoint_size_mb` | MB | Consider for storage | mean of checkpoint_size_mb |

**Why it matters:** Checkpointing is expensive; the goal is to minimize wasted time while maintaining fault tolerance.

### 2. **Training Performance** (Essential for Fairness)
Ensure all methods converge to similar quality.

| Metric | Unit | Direction | Capture From |
|--------|------|-----------|--------------|
| `best_val_accuracy` | % | Higher is better | summary.best_val_accuracy |
| `final_val_accuracy` | % | Higher is better | summary.final_val_accuracy |
| `best_train_loss` | loss | Lower is better | summary.best_train_loss |
| `epochs_to_target` | count | Fewer is better | from epoch_metrics |

**Why it matters:** All methods must achieve similar accuracy; we're comparing efficiency, not performance quality.

### 3. **Convergence Efficiency** (COCI modes only)
Measure how well the convergence scheduler adapts.

| Metric | Unit | Direction | Capture From |
|--------|------|-----------|--------------|
| `total_convergence_checkpoints` | count | Compare to total_checkpoint_count | summary.convergence_metrics.total_convergence_checkpoints |
| `current_interval_sec` | seconds | Should adapt based on phase | summary.convergence_metrics.current_interval_sec |
| `theta1` (decay) | rate | Indicates convergence speed | summary.convergence_metrics.theta1 |
| `loss_fit_quality` | error | Lower error = better fit | summary.convergence_metrics.loss_fit_error |

**Why it matters:** Validates that COCI is adapting checkpoints intelligently.

**Key ratio to compute:**
```
Selectivity Ratio = convergence_checkpoints / epoch_checkpoints
```
Higher ratio means COCI is more selective (fewer redundant checkpoints).

### 4. **Hash Ring Effectiveness** (Hash-ring modes only)
Measure cache locality and recovery efficiency.

| Metric | Unit | Direction | Capture From |
|--------|------|-----------|--------------|
| `cache_hit_rate_percent` | % | Higher is better | (cache_hits / (cache_hits + cache_misses)) * 100 |
| `shard_recovery_count` | count | Lower better (fewer failures) | summary.hash_ring_metrics.shard_recovery_count |
| `cache_size_mb` / `max_cache_mb` | % | Lower overhead | cache_size_mb / max_cache_mb * 100 |
| `local_cache_hits` | count | More locality = better | summary.hash_ring_metrics.local_cache_hits |

**Why it matters:** Validates that hash-ring sharding provides good cache locality.

**Key ratio to compute:**
```
Cache Utilization = cache_size_mb / max_cache_mb
Cache Efficiency = cache_hit_rate * (1 - cache_utilization)
```

### 5. **Total Training Time** (Overall Metric)
End-to-end comparison including all overheads.

| Metric | Unit | Direction | Capture From |
|--------|------|-----------|--------------|
| `total_training_time_sec` | seconds | Lower is better | summary.total_training_time_sec |
| `samples_per_sec` | throughput | Higher is better | total_samples / total_training_time |
| `comm_overhead_percent` | % | Lower is better | summary.distributed_metrics.comm_overhead_percent |

**Why it matters:** Ultimate goal is faster training with same quality.

### 6. **Fault Tolerance** (Optional but Important)
If running fault injection experiments:

| Metric | Unit | Direction | Capture From |
|--------|------|-----------|--------------|
| `recovery_success_rate` | % | Higher is better | recovered_successfully / (recovered_successfully + recovery_failures) * 100 |
| `total_recovery_time_sec` | seconds | Lower is better | summary.fault_metrics.total_recovery_time_sec |
| `injected_faults` | count | For validation | summary.fault_metrics.injected_faults |

**Why it matters:** Validates fault tolerance strategy effectiveness.

## Comparison Table Template

Use this table to compare results:

```markdown
| Metric | Epoch | Convergence | Hash-Ring-Epoch | Conv-Hash-Ring |
|--------|-------|-------------|-----------------|----------------|
| **Checkpoint Efficiency** | | | | |
| Checkpoint Overhead (%) | 12.5 | 8.2 | 11.8 | 7.5 |
| Total Checkpoint Time (s) | 450 | 290 | 425 | 265 |
| Checkpoint Count | 20 | 15 | 20 | 14 |
| **Training Performance** | | | | |
| Best Val Accuracy (%) | 92.50 | 92.48 | 92.52 | 92.49 |
| Final Val Accuracy (%) | 91.80 | 91.95 | 91.75 | 92.05 |
| Best Train Loss | 0.0234 | 0.0198 | 0.0245 | 0.0189 |
| **Convergence Metrics (COCI)** | | | | |
| Conv Checkpoints | - | 15 | - | 14 |
| Avg Interval (s) | - | 287.3 | - | 285.6 |
| Theta1 (decay) | - | -0.00234 | - | -0.00241 |
| **Hash Ring Metrics** | | | | |
| Cache Hit Rate (%) | - | - | 68.5 | 72.3 |
| Shards Owned | - | - | 12 | 12 |
| Cache Size (MB) | - | - | 2845 | 2923 |
| **Overall** | | | | |
| Total Training Time (s) | 3600 | 3537 | 3598 | 3512 |
| Speedup vs Epoch | 1.0x | 1.02x | 1.00x | 1.02x |
```

## Statistical Comparison (Multiple Runs)

When running each mode multiple times (recommended 3-5 runs):

```python
import pandas as pd
from scipy import stats

# Load JSONL results
results = pd.read_json("experiment_results/experiment_summaries.jsonl", lines=True)

# Filter by mode and experiment
for mode in ['epoch', 'convergence', 'hash-ring-epoch', 'convergence-hash-ring']:
    data = results[results['training_mode'] == mode]
    
    print(f"\n{mode}:")
    print(f"  Training Time: {data['total_training_time_sec'].mean():.1f}s ± {data['total_training_time_sec'].std():.1f}s")
    print(f"  Checkpoint Overhead: {data['checkpoint_overhead_percent'].mean():.2f}% ± {data['checkpoint_overhead_percent'].std():.2f}%")
    print(f"  Best Val Acc: {data['best_val_accuracy'].mean():.2f}% ± {data['best_val_accuracy'].std():.2f}%")
```

## Visualization Recommendations

### Plot 1: Checkpoint Overhead Comparison
```
x-axis: Training Mode
y-axis: Checkpoint Overhead %
type: Bar chart
legend: Show error bars for std dev (if multiple runs)
```

### Plot 2: Training Time Comparison
```
x-axis: Training Mode
y-axis: Total Training Time (seconds)
type: Bar chart with speedup overlay
reference: epoch mode = 1.0x baseline
```

### Plot 3: Loss/Accuracy Convergence
```
x-axis: Epoch
y-axis: Validation Loss / Accuracy
type: Line plot, one line per mode
note: Overlay all modes to verify similar convergence
```

### Plot 4: COCI Interval Adaptation (COCI modes only)
```
x-axis: Time (seconds)
y-axis: Checkpoint Interval (seconds)
type: Line plot
note: Should show adaptive behavior, not flat like epoch mode
```

### Plot 5: Hash Ring Cache Hit Rate (Hash-ring modes only)
```
x-axis: Time or Epoch
y-axis: Cumulative Cache Hit Rate (%)
type: Line plot
note: Should show increasing hit rate as cache fills
```

## Reporting Results

Provide these sections when comparing methods:

1. **Executive Summary**
   - Which method is fastest? By how much?
   - Which method is most reliable?
   - Recommended method for your use case?

2. **Efficiency Metrics**
   - Table with checkpoint overhead for all methods
   - Speedup ratios compared to baseline (epoch)

3. **Quality Assurance**
   - All methods achieve similar accuracy?
   - Convergence curves are aligned?
   - Any method diverges or fails?

4. **Deep Dive (per mode)**
   - Convergence: How adaptive is the scheduling?
   - Hash Ring: Cache effectiveness?
   - Faults: Recovery success rate?

5. **Scalability** (if distributed)
   - How do methods scale to more GPUs?
   - Communication overhead trends?

## Quick Checklist

Before comparing, ensure:

- [ ] All methods trained for **same number of epochs**
- [ ] **Same hyperparameters** (lr, batch size, optimizer)
- [ ] **Same dataset** split and augmentation
- [ ] All methods achieve **similar validation accuracy** (within 1%)
- [ ] Multiple runs done (3-5 minimum for statistical significance)
- [ ] Results saved in **JSONL format** for easy aggregation
- [ ] Visualizations use **consistent scales** across modes

## Example Commands

```bash
# Run all four modes
python scripts/train_faceforensics.py --training-mode epoch --epochs 20
python scripts/train_faceforensics.py --training-mode convergence --epochs 20
python scripts/train_faceforensics.py --training-mode hash-ring-epoch --epochs 20
python scripts/train_faceforensics.py --training-mode convergence-hash-ring --epochs 20

# Export and compare
python -c "
from src.coci.metrics import MetricsExporter
import pandas as pd

exporter = MetricsExporter()
results = exporter.load_jsonl_results('experiment_results/experiment_summaries.jsonl')
df = pd.DataFrame(results)

comparison = exporter.compare_modes(results)
for mode, stats in comparison.items():
    print(f'{mode}: {stats[\"avg_training_time_sec\"]:.1f}s, {stats[\"avg_checkpoint_overhead_percent\"]:.2f}% overhead')
"
```

## See Also

- `METRICS_GUIDE.md` - Detailed integration guide
- [Integration Instructions](#integration-into-training-scripts) - How to add metrics to scripts
