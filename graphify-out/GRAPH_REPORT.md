# Graph Report - D:\Sai\HPC\project\adaptive-ring-checkpointing  (2026-04-19)

## Corpus Check
- 29 files · ~64,337 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 495 nodes · 1078 edges · 54 communities detected
- Extraction: 48% EXTRACTED · 52% INFERRED · 0% AMBIGUOUS · INFERRED: 558 edges (avg confidence: 0.59)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]

## God Nodes (most connected - your core abstractions)
1. `CheckpointManager` - 60 edges
2. `MetricsCollector` - 59 edges
3. `CheckpointBitFlipInjector` - 58 edges
4. `main()` - 55 edges
5. `FaultType` - 49 edges
6. `CheckpointFaultConfig` - 47 edges
7. `HashRingCheckpointManager` - 46 edges
8. `FaceForensicsDataset` - 45 edges
9. `VideoFaceForensicsDataset` - 42 edges
10. `ConvergenceAwareScheduler` - 40 edges

## Surprising Connections (you probably didn't know these)
- `CheckpointManager` --uses--> `Return runtime metrics for resilience analysis and reporting.`  [INFERRED]
  D:\Sai\HPC\project\adaptive-ring-checkpointing\src\coci\checkpointing\checkpoint_manager.py → D:\Sai\HPC\project\adaptive-ring-checkpointing\src\coci\checkpointing\hash_ring_checkpoint_manager.py
- `example_compare_results()` --calls--> `load_jsonl_results()`  [INFERRED]
  D:\Sai\HPC\project\adaptive-ring-checkpointing\scripts\example_metrics.py → D:\Sai\HPC\project\adaptive-ring-checkpointing\src\coci\metrics\exporter.py
- `example_compare_results()` --calls--> `compare_modes()`  [INFERRED]
  D:\Sai\HPC\project\adaptive-ring-checkpointing\scripts\example_metrics.py → D:\Sai\HPC\project\adaptive-ring-checkpointing\src\coci\metrics\exporter.py
- `train()` --calls--> `train_epoch()`  [INFERRED]
  D:\Sai\HPC\project\adaptive-ring-checkpointing\scripts\train.py → D:\Sai\HPC\project\adaptive-ring-checkpointing\scripts\train_distributed.py
- `train()` --calls--> `train_epoch()`  [INFERRED]
  D:\Sai\HPC\project\adaptive-ring-checkpointing\scripts\train.py → D:\Sai\HPC\project\adaptive-ring-checkpointing\scripts\train_faceforensics.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.03
Nodes (67): ConvergenceMetrics, DistributedMetrics, EpochMetrics, ExperimentSummary, FaultMetrics, HashRingMetrics, MetricsCollector, Comprehensive metrics collection for training experiments. (+59 more)

### Community 1 - "Community 1"
Cohesion: 0.13
Nodes (64): CheckpointBitFlipInjector, CheckpointFaultConfig, FaultType, Inject bit-flips into checkpoint tensors before model/optimizer restore., CheckpointManager, ConvergenceAwareScheduler, COCI-inspired online scheduler for convergence-aware checkpoint timing.      T, Dataset (+56 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (47): enforce_determinism(), Force deterministic behavior to isolate bit-flip impact from training noise., Save a checkpoint.          Args:             model: The model to save (or DD, Record where a checkpoint was loaded from., Record detected fault., Record that a resume was attempted., Record a successful resume event and its rollback characteristics., Record the last observed failure state. (+39 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (18): Enum, FaultDetector, NodeStatus, Broadcast heartbeat signals to all peers using dist.all_reduce.          Returns, Gather suspicion votes from all ranks using dist.all_gather.          Returns ag, Start background thread for periodic heartbeat broadcast.          Args:, Stop the heartbeat thread gracefully.          Args:             timeout: Maximu, Background loop that periodically broadcasts heartbeats. (+10 more)

### Community 4 - "Community 4"
Cohesion: 0.07
Nodes (17): download_faceforensics_dataset(), precompute_face_crops(), preextract_frames(), process_batch(), FaceForensics++ dataset loader with MTCNN face detection.  Provides: - FaceFo, Process a batch of images through MTCNN., Download FaceForensics++ dataset from Kaggle via kagglehub.      Returns:, Pre-extract frames from videos and save as .npy files.      This function proc (+9 more)

### Community 5 - "Community 5"
Cohesion: 0.09
Nodes (16): ABC, get_cifar100_dataset(), Record in-epoch checkpoint metrics (e.g., convergence-based)., Config, load_config(), FaultInjector, Rank-aware fault injector for simulating GPU/node failures.      Supports two, CheckpointStrategy (+8 more)

### Community 6 - "Community 6"
Cohesion: 0.14
Nodes (10): Load the latest checkpoint and restore model/optimizer state.          Args:, export_csv_table(), export_markdown_table(), generate_full_run_rows(), generate_recovery_rows(), load_rows(), main(), save_bar_plot() (+2 more)

### Community 7 - "Community 7"
Cohesion: 0.12
Nodes (8): get_faceforensics_transforms(), Load samples from pre-computed crop cache., Set up for on-the-fly MTCNN detection (slow)., Build an index of videos from the pre-computed crops manifest.          Each v, Build video index from pre-computed crop manifest., Get transforms for FaceForensics++ face images.      Args:         include_re, Build video index from dataset directory structure., Load all .npy file paths from real/fake directories.

### Community 8 - "Community 8"
Cohesion: 0.24
Nodes (8): get_efficientnet_binary(), get_model(), get_multiframe_model(), MultiFrameModel, Multi-frame video classification model.      Processes multiple frames through, Forward pass.          Args:             x: Input tensor of shape (B, T, C, H, Create EfficientNet-B0 model for binary classification.      Args:         pr, Create a multi-frame model for video-level deepfake detection.      This model

### Community 9 - "Community 9"
Cohesion: 0.33
Nodes (1): ConvergenceFit

### Community 10 - "Community 10"
Cohesion: 0.33
Nodes (3): Get a single sample from the dataset.          If using pre-computed crops (re, Load pre-computed face crop from numpy file., Detect face using MTCNN (slow).

### Community 11 - "Community 11"
Cohesion: 0.33
Nodes (3): Load all image paths from the dataset directory., Recursively collect image paths from video folder structure., Load images from flat directory structure.

### Community 12 - "Community 12"
Cohesion: 0.33
Nodes (6): Adaptive Ring Checkpointing, Convergence-Aware Checkpointing (COCI), Soft Error Injection Mechanism, Hash Ring Checkpointing, Loss Curve Model (Exponential Decay), Piecewise Online Fitting (POF)

### Community 13 - "Community 13"
Cohesion: 0.4
Nodes (1): ImageDataset

### Community 14 - "Community 14"
Cohesion: 0.5
Nodes (2): Get frames from a single video.          Args:             idx: Index of the, Load a single frame and apply transforms.

### Community 15 - "Community 15"
Cohesion: 0.5
Nodes (4): End-to-End Fault Tolerance Verification, Phase 4 - Exception Handling, Phase 5 - Fault Injection, Phase 6 - Checkpoint Recovery

### Community 16 - "Community 16"
Cohesion: 1.0
Nodes (0): 

### Community 17 - "Community 17"
Cohesion: 1.0
Nodes (1): Return checkpoint-manager runtime metrics in a hash-ring-compatible shape.

### Community 18 - "Community 18"
Cohesion: 1.0
Nodes (1): Return the number of samples in the dataset.

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (1): Get frames from a single video.          Returns:             tuple: (frames_

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (1): Return the number of videos in the dataset.

### Community 21 - "Community 21"
Cohesion: 1.0
Nodes (0): 

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (0): 

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (0): 

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (0): 

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): Check if running in distributed mode.

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): Save a checkpoint.          Args:             model: The model to save (or DD

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): Load the latest checkpoint and restore model/optimizer state.          Args:

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (1): Fault injection and recovery metrics.

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): Distributed training metrics.

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): Overall experiment summary.

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): Collects metrics throughout training for later export and analysis.

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): Initialize metrics collector.                  Args:             experiment_n

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (1): Mark the start of an epoch.

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): Record metrics for completed epoch.                  Args:             epoch:

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): Record checkpoint metrics.                  Args:             checkpoint_size

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (1): Record in-epoch checkpoint metrics (e.g., convergence-based).

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): Update convergence-aware scheduler metrics.                  Args:

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): Increment convergence-based checkpoint counter.

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Record that convergence fitting segment restarted (POF).

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Update hash-ring checkpoint metrics.                  Args:             total

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): Record local cache hit.

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): Record local cache miss.

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): Record shard recovery event.                  Args:             recovery_time

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): Record injected fault.

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): Record detected fault.

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Record recovery attempt.                  Args:             success: Whether

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): Record checkpoint integrity check failure.

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): Set distributed training configuration.                  Args:             wo

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Record collective synchronization operation.                  Args:

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Increment batch counter.                  Args:             samples: Number o

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Generate final summary metrics.

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): Get all epoch metrics as dictionaries for export.

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): Get current state snapshot for debugging.

## Knowledge Gaps
- **132 isolated node(s):** `Example of training loop with metrics collection.`, `Example of comparing results across modes.`, `Distributed training utilities for multi-GPU coordination.  Provides functions`, `Check if the default process group is initialized.      Returns:         bool`, `Get the rank of the current process in the distributed group.      Returns:` (+127 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 16`** (2 nodes): `main()`, `main.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 17`** (2 nodes): `.get_runtime_metrics()`, `Return checkpoint-manager runtime metrics in a hash-ring-compatible shape.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 18`** (2 nodes): `.__len__()`, `Return the number of samples in the dataset.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (2 nodes): `Get frames from a single video.          Returns:             tuple: (frames_`, `.__getitem__()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (2 nodes): `Return the number of videos in the dataset.`, `.__len__()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (1 nodes): `train_convergence_hashring.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `train_convergence_normal.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `train_epoch_based.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `train_hashring_epoch.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `Check if running in distributed mode.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `Save a checkpoint.          Args:             model: The model to save (or DD`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `Load the latest checkpoint and restore model/optimizer state.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `Fault injection and recovery metrics.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `Distributed training metrics.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `Overall experiment summary.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Collects metrics throughout training for later export and analysis.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `Initialize metrics collector.                  Args:             experiment_n`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `Mark the start of an epoch.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `Record metrics for completed epoch.                  Args:             epoch:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `Record checkpoint metrics.                  Args:             checkpoint_size`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `Record in-epoch checkpoint metrics (e.g., convergence-based).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `Update convergence-aware scheduler metrics.                  Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Increment convergence-based checkpoint counter.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Record that convergence fitting segment restarted (POF).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Update hash-ring checkpoint metrics.                  Args:             total`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Record local cache hit.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Record local cache miss.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Record shard recovery event.                  Args:             recovery_time`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `Record injected fault.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Record detected fault.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Record recovery attempt.                  Args:             success: Whether`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `Record checkpoint integrity check failure.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `Set distributed training configuration.                  Args:             wo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Record collective synchronization operation.                  Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Increment batch counter.                  Args:             samples: Number o`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Generate final summary metrics.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Get all epoch metrics as dictionaries for export.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Get current state snapshot for debugging.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Community 2` to `Community 0`, `Community 1`, `Community 3`, `Community 4`, `Community 6`, `Community 7`, `Community 8`?**
  _High betweenness centrality (0.399) - this node is a cross-community bridge._
- **Why does `MetricsCollector` connect `Community 0` to `Community 2`, `Community 5`?**
  _High betweenness centrality (0.217) - this node is a cross-community bridge._
- **Why does `main()` connect `Community 5` to `Community 0`, `Community 1`, `Community 2`, `Community 4`, `Community 8`?**
  _High betweenness centrality (0.095) - this node is a cross-community bridge._
- **Are the 54 inferred relationships involving `CheckpointManager` (e.g. with `Distributed training entry point using torchrun.  This script is designed to b` and `Seed worker for reproducible data loading across epochs.     Ensures each worke`) actually correct?**
  _`CheckpointManager` has 54 INFERRED edges - model-reasoned connections that need verification._
- **Are the 26 inferred relationships involving `MetricsCollector` (e.g. with `MetricsExporter` and `Export metrics to various formats for analysis and plotting.`) actually correct?**
  _`MetricsCollector` has 26 INFERRED edges - model-reasoned connections that need verification._
- **Are the 46 inferred relationships involving `CheckpointBitFlipInjector` (e.g. with `Distributed training entry point using torchrun.  This script is designed to b` and `Seed worker for reproducible data loading across epochs.     Ensures each worke`) actually correct?**
  _`CheckpointBitFlipInjector` has 46 INFERRED edges - model-reasoned connections that need verification._
- **Are the 47 inferred relationships involving `main()` (e.g. with `log_on_main()` and `preextract_frames()`) actually correct?**
  _`main()` has 47 INFERRED edges - model-reasoned connections that need verification._