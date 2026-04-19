# Graph Report - D:\Sai\HPC\project\adaptive-ring-checkpointing  (2026-04-19)

## Corpus Check
- 28 files · ~36,540 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 411 nodes · 906 edges · 25 communities detected
- Extraction: 51% EXTRACTED · 49% INFERRED · 0% AMBIGUOUS · INFERRED: 440 edges (avg confidence: 0.61)
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

## God Nodes (most connected - your core abstractions)
1. `CheckpointBitFlipInjector` - 48 edges
2. `CheckpointManager` - 46 edges
3. `main()` - 45 edges
4. `MetricsCollector` - 43 edges
5. `FaultType` - 39 edges
6. `CheckpointFaultConfig` - 37 edges
7. `HashRingCheckpointManager` - 35 edges
8. `FaceForensicsDataset` - 35 edges
9. `main()` - 32 edges
10. `VideoFaceForensicsDataset` - 32 edges

## Surprising Connections (you probably didn't know these)
- `example_compare_results()` --calls--> `load_jsonl_results()`  [INFERRED]
  D:\Sai\HPC\project\adaptive-ring-checkpointing\scripts\example_metrics.py → D:\Sai\HPC\project\adaptive-ring-checkpointing\src\coci\metrics\exporter.py
- `example_compare_results()` --calls--> `compare_modes()`  [INFERRED]
  D:\Sai\HPC\project\adaptive-ring-checkpointing\scripts\example_metrics.py → D:\Sai\HPC\project\adaptive-ring-checkpointing\src\coci\metrics\exporter.py
- `train()` --calls--> `train_epoch()`  [INFERRED]
  D:\Sai\HPC\project\adaptive-ring-checkpointing\scripts\train.py → D:\Sai\HPC\project\adaptive-ring-checkpointing\scripts\train_distributed.py
- `train()` --calls--> `train_epoch()`  [INFERRED]
  D:\Sai\HPC\project\adaptive-ring-checkpointing\scripts\train.py → D:\Sai\HPC\project\adaptive-ring-checkpointing\scripts\train_faceforensics.py
- `main()` --calls--> `load_config()`  [INFERRED]
  D:\Sai\HPC\project\adaptive-ring-checkpointing\scripts\train.py → src\coci\config.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.04
Nodes (58): get_cifar100_dataset(), ConvergenceMetrics, DistributedMetrics, EpochMetrics, ExperimentSummary, FaultMetrics, HashRingMetrics, MetricsCollector (+50 more)

### Community 1 - "Community 1"
Cohesion: 0.11
Nodes (54): CheckpointBitFlipInjector, CheckpointFaultConfig, FaultType, Inject bit-flips into checkpoint tensors before model/optimizer restore., CheckpointManager, ConvergenceAwareScheduler, COCI-inspired online scheduler for convergence-aware checkpoint timing.      T, Dataset (+46 more)

### Community 2 - "Community 2"
Cohesion: 0.07
Nodes (40): enforce_determinism(), Force deterministic behavior to isolate bit-flip impact from training noise., Save a checkpoint.          Args:             model: The model to save (or DD, Mark the start of an epoch., Record checkpoint metrics.                  Args:             checkpoint_size, barrier(), cleanup_distributed(), get_local_rank() (+32 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (17): Enum, FaultDetector, NodeStatus, Broadcast heartbeat signals to all peers using dist.all_reduce.          Returns, Gather suspicion votes from all ranks using dist.all_gather.          Returns ag, Start background thread for periodic heartbeat broadcast.          Args:, Stop the heartbeat thread gracefully.          Args:             timeout: Maximu, Background loop that periodically broadcasts heartbeats. (+9 more)

### Community 4 - "Community 4"
Cohesion: 0.08
Nodes (7): ElasticRecaching, HashRing, Write shard to local NVMe cache with atomic rename pattern., Load shard from local cache with verification., Verify shard integrity by loading and checking., Remove cached files for shards owned by a dead node., ShardManager

### Community 5 - "Community 5"
Cohesion: 0.1
Nodes (12): ABC, Record in-epoch checkpoint metrics (e.g., convergence-based)., FaultInjector, Rank-aware fault injector for simulating GPU/node failures.      Supports two, CheckpointStrategy, CheckpointStrategyFactory, create(), EpochStrategy (+4 more)

### Community 6 - "Community 6"
Cohesion: 0.17
Nodes (3): Load the latest checkpoint and restore model/optimizer state.          Args:, Return cache usage statistics., str

### Community 7 - "Community 7"
Cohesion: 0.24
Nodes (8): get_efficientnet_binary(), get_model(), get_multiframe_model(), MultiFrameModel, Multi-frame video classification model.      Processes multiple frames through, Forward pass.          Args:             x: Input tensor of shape (B, T, C, H, Create EfficientNet-B0 model for binary classification.      Args:         pr, Create a multi-frame model for video-level deepfake detection.      This model

### Community 8 - "Community 8"
Cohesion: 0.22
Nodes (9): download_faceforensics_dataset(), precompute_face_crops(), preextract_frames(), process_batch(), FaceForensics++ dataset loader with MTCNN face detection.  Provides: - FaceFo, Process a batch of images through MTCNN., Download FaceForensics++ dataset from Kaggle via kagglehub.      Returns:, Pre-extract frames from videos and save as .npy files.      This function proc (+1 more)

### Community 9 - "Community 9"
Cohesion: 0.25
Nodes (3): Load samples from pre-computed crop cache., Set up for on-the-fly MTCNN detection (slow)., Load all .npy file paths from real/fake directories.

### Community 10 - "Community 10"
Cohesion: 0.29
Nodes (3): Build an index of videos from the pre-computed crops manifest.          Each v, Build video index from pre-computed crop manifest., Build video index from dataset directory structure.

### Community 11 - "Community 11"
Cohesion: 0.33
Nodes (1): ConvergenceFit

### Community 12 - "Community 12"
Cohesion: 0.33
Nodes (3): Load all image paths from the dataset directory., Recursively collect image paths from video folder structure., Load images from flat directory structure.

### Community 13 - "Community 13"
Cohesion: 0.33
Nodes (3): Get a single sample from the dataset.          If using pre-computed crops (re, Load pre-computed face crop from numpy file., Detect face using MTCNN (slow).

### Community 14 - "Community 14"
Cohesion: 0.33
Nodes (6): Adaptive Ring Checkpointing, Convergence-Aware Checkpointing (COCI), Soft Error Injection Mechanism, Hash Ring Checkpointing, Loss Curve Model (Exponential Decay), Piecewise Online Fitting (POF)

### Community 15 - "Community 15"
Cohesion: 0.4
Nodes (1): ImageDataset

### Community 16 - "Community 16"
Cohesion: 0.5
Nodes (2): Get frames from a single video.          Args:             idx: Index of the, Load a single frame and apply transforms.

### Community 17 - "Community 17"
Cohesion: 0.5
Nodes (4): End-to-End Fault Tolerance Verification, Phase 4 - Exception Handling, Phase 5 - Fault Injection, Phase 6 - Checkpoint Recovery

### Community 18 - "Community 18"
Cohesion: 1.0
Nodes (2): Config, load_config()

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (0): 

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (0): 

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
Nodes (1): Check if running in distributed mode.

## Knowledge Gaps
- **92 isolated node(s):** `Example of training loop with metrics collection.`, `Example of comparing results across modes.`, `Distributed training utilities for multi-GPU coordination.  Provides functions`, `Check if the default process group is initialized.      Returns:         bool`, `Get the rank of the current process in the distributed group.      Returns:` (+87 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 19`** (2 nodes): `main()`, `main.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `train_convergence_hashring.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (1 nodes): `train_convergence_normal.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `train_epoch_based.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `train_hashring_epoch.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `Check if running in distributed mode.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Community 2` to `Community 0`, `Community 1`, `Community 3`, `Community 4`, `Community 6`, `Community 7`, `Community 8`?**
  _High betweenness centrality (0.433) - this node is a cross-community bridge._
- **Why does `MetricsCollector` connect `Community 0` to `Community 2`, `Community 5`?**
  _High betweenness centrality (0.222) - this node is a cross-community bridge._
- **Why does `main()` connect `Community 0` to `Community 1`, `Community 2`, `Community 4`, `Community 5`, `Community 7`, `Community 18`?**
  _High betweenness centrality (0.115) - this node is a cross-community bridge._
- **Are the 36 inferred relationships involving `CheckpointBitFlipInjector` (e.g. with `Distributed training entry point using torchrun.  This script is designed to b` and `Seed worker for reproducible data loading across epochs.     Ensures each worke`) actually correct?**
  _`CheckpointBitFlipInjector` has 36 INFERRED edges - model-reasoned connections that need verification._
- **Are the 41 inferred relationships involving `CheckpointManager` (e.g. with `Distributed training entry point using torchrun.  This script is designed to b` and `Seed worker for reproducible data loading across epochs.     Ensures each worke`) actually correct?**
  _`CheckpointManager` has 41 INFERRED edges - model-reasoned connections that need verification._
- **Are the 40 inferred relationships involving `main()` (e.g. with `log_on_main()` and `preextract_frames()`) actually correct?**
  _`main()` has 40 INFERRED edges - model-reasoned connections that need verification._
- **Are the 19 inferred relationships involving `MetricsCollector` (e.g. with `MetricsExporter` and `Export metrics to various formats for analysis and plotting.`) actually correct?**
  _`MetricsCollector` has 19 INFERRED edges - model-reasoned connections that need verification._