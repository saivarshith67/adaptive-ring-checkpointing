# Experiment Report: faceforensics_convergence_hash_ring_20260419_140224

**Training Mode:** convergence-hash-ring  
**Dataset:** faceforensics  
**Model:** efficientnet_b0  
**Configuration:** 20 epochs, batch size 2  
**Generated:** 2026-04-19 14:10:22

## Training Performance

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | 73.75% |
| Final Validation Accuracy | 71.25% |
| Best Validation Loss | 0.7370 |
| Final Validation Loss | 0.8699 |
| Best Training Loss | 0.2504 |
| Final Training Loss | 0.2584 |
| Total Training Time | 477.6s |

## Checkpointing Overhead

| Metric | Value |
|--------|-------|
| Total Checkpoint Count | 23 |
| Total Checkpoint Time | 21.1s |
| Checkpoint Overhead | 4.4% |

## Convergence Scheduler Metrics

| Metric | Value |
|--------|-------|
| Current Interval (seconds) | 1800.00 |
| Total Convergence Checkpoints | 18 |
| Loss Fit Segments Restarted | 0 |
| Theta1 (Exponential Decay) | -0.0012031961098375335 |
| Theta2 (Exponential Offset) | -1.723258394591353 |

## Hash Ring Metrics

| Metric | Value |
|--------|-------|
| Local Cache Hits | 0 |
| Local Cache Misses | 0 |
| Cache Size (MB) | 463.4 |
| Max Cache Size (MB) | 463.4 |
| Total Shards | 23 |
| Shards Owned by This Node | 4 |
| Central Storage Loads | 0 |
| Local Cache Loads | 0 |
| Cache Writes | 4 |
| Orphaned Shards | 0 |
| Reassigned Shards | 0 |
| Re-cached Shards | 0 |
| Shard Recovery Count | 0 |
| Total Recovery Time | 0.0s |
| Last Load Source | central-storage |

## Fault Tolerance Metrics

| Metric | Value |
|--------|-------|
| Injected Faults | 0 |
| Detected Faults | 0 |
| Recovery Attempts | 0 |
| Successfully Recovered | 0 |
| Recovery Failures | 0 |
| Checkpoint Integrity Failures | 0 |
| Total Recovery Time | 0.0s |
| Runtime Fault Triggered | False |
| Runtime Fault Checkpoint | N/A |
| Runtime Fault Epoch | N/A |
| Runtime Fault Global Step | N/A |
| Resume Attempted | False |
| Resume Succeeded | False |
| Resumed From Epoch | N/A |
| Resumed Checkpoint | N/A |
| Resume Load Source | N/A |
| Time to Resume | 0.0s |
| Rollback Epochs | 0.00 |
| Rollback Steps | 0 |
| Lost Work Time | 0.0s |

## Distributed Training Metrics

| Metric | Value |
|--------|-------|
| World Size | 4 |
| Total Synchronization Operations | 0 |
| Total Synchronization Time | 0.0s |
| Communication Overhead | 0.0% |
