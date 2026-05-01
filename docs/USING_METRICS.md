# Quick Start: Using the New Metrics System

**TL;DR:** All training scripts now automatically capture metrics. Just run them normally!

---

## How It Works

1. **Start Training**: Run any training script as usual
2. **Metrics Captured Automatically**: No special flags or configuration needed
3. **Metrics Exported at End**: JSON, CSV, HTML, and Markdown files created automatically
4. **Results in `experiment_results/`**: Check the directory after training completes

---

## Run Training

### FaceForensics++ (All 4 Modes)

```bash
# Mode 1: Epoch-based checkpointing
python scripts/train_epoch_based.py --dataset-path ./data/faceforensics/FF++

# Mode 2: Convergence-aware checkpointing (COCI)
python scripts/train_convergence_normal.py --dataset-path ./data/faceforensics/FF++

# Mode 3: Hash-ring-epoch checkpointing
python scripts/train_hashring_epoch.py --dataset-path ./data/faceforensics/FF++

# Mode 4: Convergence + Hash-ring checkpointing
python scripts/train_convergence_hashring.py --dataset-path ./data/faceforensics/FF++
```

### CIFAR100 Single-GPU

```bash
python scripts/train.py --mode dev
```

### Distributed Multi-GPU

```bash
# 2 GPUs
torchrun --nproc_per_node=2 scripts/train_distributed.py

# 4 GPUs
torchrun --nproc_per_node=4 scripts/train_distributed.py

# Multiple nodes (example: 2 nodes, 4 GPUs each)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=<master_ip> --master_port=29500 scripts/train_distributed.py
```

---

## Check Results

After training completes, you'll see:

```
Exporting metrics...
  ✓ JSON summary: experiment_results/faceforensics_convergence_hash_ring_20260418_103000/convergence-hash-ring_20260418_103000.json
  ✓ JSONL aggregated: experiment_results/experiment_summaries.jsonl
  ✓ Epoch CSV: experiment_results/faceforensics_convergence_hash_ring_20260418_103000/convergence-hash-ring_epochs_20260418_103000.csv
  ✓ HTML report: experiment_results/faceforensics_convergence_hash_ring_20260418_103000/convergence-hash-ring_20260418_103000.html
  ✓ Markdown report: experiment_results/faceforensics_convergence_hash_ring_20260418_103000/convergence-hash-ring_20260418_103000.md

Metrics saved to: ./experiment_results/faceforensics_convergence_hash_ring_20260418_103000/
```

### View Results

```bash
# List all runs
ls -la experiment_results/

# View specific run
ls -la experiment_results/faceforensics_convergence_hash_ring_20260418_103000/

# View aggregated results (across all runs)
head experiment_results/experiment_summaries.jsonl
```

---

## Compare Methods

### Generate Comparison Report

```python
from src.coci.metrics import MetricsExporter
import json

exporter = MetricsExporter()

# Load all runs
results = exporter.load_jsonl_results("./experiment_results/experiment_summaries.jsonl")

# Compare modes
comparison = exporter.compare_modes(results)

# Pretty print
for mode, stats in comparison.items():
    print(f"\n{mode.upper()}")
    print(f"  Runs: {stats['num_runs']}")
    print(f"  Avg Training Time: {stats['avg_training_time_sec']:.1f}s")
    print(f"  Avg Checkpoint Overhead: {stats['avg_checkpoint_overhead_percent']:.2f}%")
    print(f"  Avg Best Val Acc: {stats['avg_best_val_accuracy']:.2f}%")
    print(f"  Avg Total Checkpoints: {stats['avg_total_checkpoint_count']:.1f}")
```

### Plot Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load epoch data from CSV
df = pd.read_csv("experiment_results/faceforensics_epoch_based_20260418_100000/epoch-based_epochs_20260418_100000.csv")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss curves
axes[0].plot(df['epoch'], df['train_loss'], marker='o', label='Train')
axes[0].plot(df['epoch'], df['val_loss'], marker='s', label='Val')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Convergence')
axes[0].legend()
axes[0].grid(True)

# Accuracy curves
axes[1].plot(df['epoch'], df['train_accuracy'], marker='o', label='Train')
axes[1].plot(df['epoch'], df['val_accuracy'], marker='s', label='Val')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy Convergence')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('convergence_comparison.png')
plt.show()
```

---

## Export Formats Explained

| Format | File | Use | Open With |
|--------|------|-----|-----------|
| **JSON** | `{mode}_YYYYMMDD_HHMMSS.json` | Detailed single-run data | Text editor / Python json.load() |
| **JSONL** | `experiment_summaries.jsonl` | Aggregate all runs | tail, pandas, or Python script |
| **CSV** | `{mode}_epochs_YYYYMMDD_HHMMSS.csv` | Plot with matplotlib/Excel | Excel, pandas, matplotlib |
| **HTML** | `{mode}_YYYYMMDD_HHMMSS.html` | Quick visual review | Browser (double-click) |
| **Markdown** | `{mode}_YYYYMMDD_HHMMSS.md` | Git-friendly documentation | Text editor / GitHub |

---

## Key Metrics to Compare

| Metric | Formula | What It Means |
|--------|---------|---------------|
| **Checkpoint Overhead %** | (total_checkpoint_time / total_training_time) × 100 | % of training time wasted checkpointing |
| **Checkpoint Count** | num_checkpoints | How many times data was saved |
| **Training Efficiency** | (useful_training_time / total_time) × 100 | Opposite of overhead |
| **Convergence Speed** | time to reach X% accuracy | How fast the model improves |
| **Final Accuracy** | best_val_accuracy | Quality of trained model |
| **Checkpoint Size** | checkpoint_size_mb | Storage requirements per checkpoint |

---

## Example: Compare All 4 Modes

```bash
# Run each mode for a few epochs
python scripts/train_epoch_based.py --epochs 3 --dataset-path ./data/faceforensics/FF++
python scripts/train_convergence_normal.py --epochs 3 --dataset-path ./data/faceforensics/FF++
python scripts/train_hashring_epoch.py --epochs 3 --dataset-path ./data/faceforensics/FF++
python scripts/train_convergence_hashring.py --epochs 3 --dataset-path ./data/faceforensics/FF++
```

Then analyze:

```python
from src.coci.metrics import MetricsExporter

exporter = MetricsExporter()
results = exporter.load_jsonl_results("./experiment_results/experiment_summaries.jsonl")
comparison = exporter.compare_modes(results)

print("COMPARISON SUMMARY")
print("=" * 60)
for mode, stats in comparison.items():
    print(f"\n{mode}")
    print(f"  Training Time: {stats['avg_training_time_sec']:.1f}s")
    print(f"  Checkpoint Overhead: {stats['avg_checkpoint_overhead_percent']:.2f}%")
    print(f"  Best Accuracy: {stats['avg_best_val_accuracy']:.2f}%")
    print(f"  Checkpoints Saved: {stats['avg_total_checkpoint_count']:.1f}")
```

---

## Troubleshooting

**Q: I don't see any metrics files**
A: Check `./experiment_results/` directory. They're created at training end. If training crashed, check the error message.

**Q: Export failed but training completed**
A: Training is fine. The metrics were collected in memory but export failed (likely permission issue). You can re-export manually:
```python
from src.coci.metrics import MetricsExporter
# Load from disk and re-export
```

**Q: How do I customize export location?**
A: Modify the MetricsExporter initialization in the script:
```python
exporter = MetricsExporter(base_export_dir="/custom/path")
```

**Q: Can I disable metrics collection?**
A: Yes, but not recommended. The system has minimal overhead. If needed, comment out the metrics export block in the script.

---

## What Gets Tracked

- ✅ **Training metrics**: loss, accuracy per epoch
- ✅ **Checkpoint metrics**: count, size, save time
- ✅ **Model info**: dataset, model name, hyperparameters
- ✅ **Mode-specific**: convergence intervals, hash-ring cache hits, distributed sync count
- ✅ **Timing info**: total training time, checkpoint overhead
- ✅ **Run metadata**: timestamp, git info (if available)

---

## Documentation

For more details, see:

- [METRICS_QUICKSTART.md](METRICS_QUICKSTART.md) - 5-minute overview
- [METRICS_GUIDE.md](METRICS_GUIDE.md) - Complete integration guide  
- [COMPARISON_METRICS.md](COMPARISON_METRICS.md) - Recommended metrics for comparison
- [INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md) - What changed and how to use it
- [scripts/example_metrics.py](scripts/example_metrics.py) - Working code example

---

## Files Modified

✅ `scripts/train_faceforensics.py` - Main FaceForensics training  
✅ `scripts/train.py` - CIFAR100 training  
✅ `scripts/train_distributed.py` - Multi-GPU distributed training  

❌ `scripts/train_*_based.py` - No changes (wrappers)  
❌ `scripts/run_*.sh` - No changes (bash runners)  

---

## Next Steps

1. **Run a training session** with any mode (2-3 epochs is enough for testing)
2. **Check results** in `experiment_results/` directory
3. **Open HTML report** in browser to see styled metrics visualization
4. **Load JSONL file** to compare across multiple runs
5. **Generate plots** from CSV data for reports

That's it! The system is fully integrated and ready to use. Just run training normally and metrics are captured automatically.
