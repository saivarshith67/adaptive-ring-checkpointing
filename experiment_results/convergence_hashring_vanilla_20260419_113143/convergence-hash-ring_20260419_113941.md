# Experiment Report: faceforensics_convergence_hash_ring_20260419_113143

**Training Mode:** convergence-hash-ring  
**Dataset:** faceforensics  
**Model:** efficientnet_b0  
**Configuration:** 20 epochs, batch size 2  
**Generated:** 2026-04-19 11:39:41

## Training Performance

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | 73.75% |
| Final Validation Accuracy | 71.25% |
| Best Validation Loss | 0.7375 |
| Final Validation Loss | 0.8712 |
| Best Training Loss | 0.2503 |
| Final Training Loss | 0.2581 |
| Total Training Time | 478.0s |

## Checkpointing Overhead

| Metric | Value |
|--------|-------|
| Total Checkpoint Count | 22 |
| Total Checkpoint Time | 26.4s |
| Checkpoint Overhead | 5.5% |

## Convergence Scheduler Metrics

| Metric | Value |
|--------|-------|
| Current Interval (seconds) | 1800.00 |
| Total Convergence Checkpoints | 17 |
| Loss Fit Segments Restarted | 0 |
| Theta1 (Exponential Decay) | -0.0012353967050475199 |
| Theta2 (Exponential Offset) | -1.7189165737924301 |

## Hash Ring Metrics

| Metric | Value |
|--------|-------|
| Local Cache Hits | 0 |
| Local Cache Misses | 0 |
| Cache Size (MB) | 0.0 |
| Max Cache Size (MB) | 0.0 |
| Total Shards | 0 |
| Shards Owned by This Node | 0 |
| Shard Recovery Count | 0 |
| Total Recovery Time | 0.0s |

## Fault Tolerance Metrics

| Metric | Value |
|--------|-------|
| Injected Faults | 1 |
| Detected Faults | 0 |
| Successfully Recovered | 0 |
| Recovery Failures | 0 |
| Checkpoint Integrity Failures | 0 |
| Total Recovery Time | 0.0s |

## Distributed Training Metrics

| Metric | Value |
|--------|-------|
| World Size | 4 |
| Total Synchronization Operations | 0 |
| Total Synchronization Time | 0.0s |
| Communication Overhead | 0.0% |
