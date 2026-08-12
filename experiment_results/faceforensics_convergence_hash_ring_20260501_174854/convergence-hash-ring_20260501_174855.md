# Experiment Report: faceforensics_convergence_hash_ring_20260501_174854

**Training Mode:** convergence-hash-ring  
**Dataset:** faceforensics  
**Model:** efficientnet_b0  
**Configuration:** 20 epochs, batch size 2  
**Generated:** 2026-05-01 17:48:55

## Training Performance

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | 0.00% |
| Final Validation Accuracy | 0.00% |
| Best Validation Loss | inf |
| Final Validation Loss | 0.0000 |
| Best Training Loss | inf |
| Final Training Loss | 0.0000 |
| Total Training Time | 0.4s |

## Checkpointing Overhead

| Metric | Value |
|--------|-------|
| Total Checkpoint Count | 0 |
| Total Checkpoint Time | 0.0s |
| Checkpoint Overhead | 0.0% |
| Checkpoint Backend | normal |
| Framework-Native Saves | 0 |
| Framework-Native Loads | 0 |
| Artifacts Logged | 0 |
| Framework Delegated | False |
| Last Checkpoint Path | N/A |

## Convergence Scheduler Metrics

| Metric | Value |
|--------|-------|
| Current Interval (seconds) | 0.00 |
| Total Convergence Checkpoints | 0 |
| Loss Fit Segments Restarted | 0 |
| Theta1 (Exponential Decay) | N/A |
| Theta2 (Exponential Offset) | N/A |

## Hash Ring Metrics

| Metric | Value |
|--------|-------|
| Local Cache Hits | 0 |
| Local Cache Misses | 0 |
| Cache Size (MB) | 4772.6 |
| Max Cache Size (MB) | 4772.6 |
| Total Shards | 1 |
| Shards Owned by This Node | 0 |
| Central Storage Loads | 1 |
| Local Cache Loads | 0 |
| Cache Writes | 0 |
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
| Detected Faults | 1 |
| Recovery Attempts | 1 |
| Successfully Recovered | 1 |
| Recovery Failures | 0 |
| Checkpoint Integrity Failures | 0 |
| Total Recovery Time | 44.5s |
| Runtime Fault Triggered | False |
| Runtime Fault Checkpoint | N/A |
| Runtime Fault Epoch | N/A |
| Runtime Fault Global Step | N/A |
| Resume Attempted | True |
| Resume Succeeded | True |
| Resumed From Epoch | 20 |
| Resumed Checkpoint | step_000000001535 |
| Resume Load Source | central-storage |
| Time to Resume | 44.5s |
| Rollback Epochs | 0.00 |
| Rollback Steps | 0 |
| Lost Work Time | 0.0s |

## Distributed Training Metrics

| Metric | Value |
|--------|-------|
| World Size | 2 |
| Total Synchronization Operations | 0 |
| Total Synchronization Time | 0.0s |
| Communication Overhead | 0.0% |
