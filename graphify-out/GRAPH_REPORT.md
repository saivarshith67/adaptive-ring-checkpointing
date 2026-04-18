# Graph Report - D:\Sai\HPC\project\adaptive-ring-checkpointing  (2026-04-18)

## Corpus Check
- 24 files · ~45,765 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 292 nodes · 538 edges · 18 communities detected
- Extraction: 67% EXTRACTED · 33% INFERRED · 0% AMBIGUOUS · INFERRED: 176 edges (avg confidence: 0.66)
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

## God Nodes (most connected - your core abstractions)
1. `main()` - 29 edges
2. `CheckpointBitFlipInjector` - 27 edges
3. `CheckpointManager` - 23 edges
4. `main()` - 21 edges
5. `FaceForensicsDataset` - 20 edges
6. `HashRingCheckpointManager` - 18 edges
7. `FaultType` - 18 edges
8. `FaultDetector` - 18 edges
9. `VideoFaceForensicsDataset` - 17 edges
10. `CheckpointFaultConfig` - 16 edges

## Surprising Connections (you probably didn't know these)
- `CheckpointManager` --uses--> `Checkpoint manager that layers hash-ring local shard caching over normal checkpo`  [INFERRED]
  D:\Sai\HPC\project\adaptive-ring-checkpointing\src\coci\checkpointing\checkpoint_manager.py → D:\Sai\HPC\project\adaptive-ring-checkpointing\src\coci\checkpointing\hash_ring_checkpoint_manager.py
- `train()` --calls--> `train_epoch()`  [INFERRED]
  D:\Sai\HPC\project\adaptive-ring-checkpointing\scripts\train.py → D:\Sai\HPC\project\adaptive-ring-checkpointing\scripts\train_distributed.py
- `train()` --calls--> `train_epoch()`  [INFERRED]
  D:\Sai\HPC\project\adaptive-ring-checkpointing\scripts\train.py → D:\Sai\HPC\project\adaptive-ring-checkpointing\scripts\train_faceforensics.py
- `main()` --calls--> `enforce_determinism()`  [INFERRED]
  D:\Sai\HPC\project\adaptive-ring-checkpointing\scripts\train.py → D:\Sai\HPC\project\adaptive-ring-checkpointing\src\coci\fault\checkpoint_fault_injector.py
- `main()` --calls--> `get_cifar100_dataset()`  [INFERRED]
  D:\Sai\HPC\project\adaptive-ring-checkpointing\scripts\train.py → src\coci\data_ingestor\cifar.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (17): Enum, FaultDetector, NodeStatus, Broadcast heartbeat signals to all peers using dist.all_reduce.          Returns, Gather suspicion votes from all ranks using dist.all_gather.          Returns ag, Start background thread for periodic heartbeat broadcast.          Args:, Stop the heartbeat thread gracefully.          Args:             timeout: Maximu, Background loop that periodically broadcasts heartbeats. (+9 more)

### Community 1 - "Community 1"
Cohesion: 0.11
Nodes (30): CheckpointBitFlipInjector, CheckpointFaultConfig, enforce_determinism(), FaultType, Force deterministic behavior to isolate bit-flip impact from training noise., Inject bit-flips into checkpoint tensors before model/optimizer restore., CheckpointManager, Load the latest checkpoint and restore model/optimizer state.          Args: (+22 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (10): ElasticRecaching, HashRingCheckpointManager, Checkpoint manager that layers hash-ring local shard caching over normal checkpo, Write shard to local NVMe cache with atomic rename pattern., Load shard from local cache with verification., Verify shard integrity by loading and checking., Return cache usage statistics., Remove cached files for shards owned by a dead node. (+2 more)

### Community 3 - "Community 3"
Cohesion: 0.1
Nodes (30): Save a checkpoint.          Args:             model: The model to save (or DD, barrier(), cleanup_distributed(), get_local_rank(), get_rank(), get_world_size(), is_distributed_initialized(), is_main_process() (+22 more)

### Community 4 - "Community 4"
Cohesion: 0.09
Nodes (15): ABC, get_cifar100_dataset(), Config, load_config(), FaultInjector, Rank-aware fault injector for simulating GPU/node failures.      Supports two, CheckpointStrategy, CheckpointStrategyFactory (+7 more)

### Community 5 - "Community 5"
Cohesion: 0.11
Nodes (12): FaceForensicsDataset, PyTorch Dataset for FaceForensics++ deepfake detection.      Supports two mode, Load samples from pre-computed crop cache., Set up for on-the-fly MTCNN detection (slow)., Load all image paths from the dataset directory., Recursively collect image paths from video folder structure., Load images from flat directory structure., Return the number of samples in the dataset. (+4 more)

### Community 6 - "Community 6"
Cohesion: 0.11
Nodes (10): Dataset, ImageDataset, Video-level PyTorch Dataset for FaceForensics++ deepfake detection.      This, Build an index of videos from the pre-computed crops manifest.          Each v, Build video index from pre-computed crop manifest., Build video index from dataset directory structure., Return the number of videos in the dataset., Get frames from a single video.          Args:             idx: Index of the (+2 more)

### Community 7 - "Community 7"
Cohesion: 0.24
Nodes (3): ConvergenceAwareScheduler, ConvergenceFit, COCI-inspired online scheduler for convergence-aware checkpoint timing.      T

### Community 8 - "Community 8"
Cohesion: 0.24
Nodes (8): get_efficientnet_binary(), get_model(), get_multiframe_model(), MultiFrameModel, Multi-frame video classification model.      Processes multiple frames through, Forward pass.          Args:             x: Input tensor of shape (B, T, C, H, Create EfficientNet-B0 model for binary classification.      Args:         pr, Create a multi-frame model for video-level deepfake detection.      This model

### Community 9 - "Community 9"
Cohesion: 0.22
Nodes (9): download_faceforensics_dataset(), precompute_face_crops(), preextract_frames(), process_batch(), FaceForensics++ dataset loader with MTCNN face detection.  Provides: - FaceFo, Process a batch of images through MTCNN., Download FaceForensics++ dataset from Kaggle via kagglehub.      Returns:, Pre-extract frames from videos and save as .npy files.      This function proc (+1 more)

### Community 10 - "Community 10"
Cohesion: 0.33
Nodes (6): Adaptive Ring Checkpointing, Convergence-Aware Checkpointing (COCI), Soft Error Injection Mechanism, Hash Ring Checkpointing, Loss Curve Model (Exponential Decay), Piecewise Online Fitting (POF)

### Community 11 - "Community 11"
Cohesion: 0.5
Nodes (4): End-to-End Fault Tolerance Verification, Phase 4 - Exception Handling, Phase 5 - Fault Injection, Phase 6 - Checkpoint Recovery

### Community 12 - "Community 12"
Cohesion: 1.0
Nodes (0): 

### Community 13 - "Community 13"
Cohesion: 1.0
Nodes (0): 

### Community 14 - "Community 14"
Cohesion: 1.0
Nodes (0): 

### Community 15 - "Community 15"
Cohesion: 1.0
Nodes (0): 

### Community 16 - "Community 16"
Cohesion: 1.0
Nodes (0): 

### Community 17 - "Community 17"
Cohesion: 1.0
Nodes (1): Check if running in distributed mode.

## Knowledge Gaps
- **60 isolated node(s):** `Distributed training utilities for multi-GPU coordination.  Provides functions`, `Check if the default process group is initialized.      Returns:         bool`, `Get the rank of the current process in the distributed group.      Returns:`, `Get the total number of processes in the distributed group.      Returns:`, `Get the local rank of the current process.      The local rank is the GPU devi` (+55 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 12`** (2 nodes): `main()`, `main.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 13`** (1 nodes): `train_convergence_hashring.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 14`** (1 nodes): `train_convergence_normal.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 15`** (1 nodes): `train_epoch_based.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 16`** (1 nodes): `train_hashring_epoch.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 17`** (1 nodes): `Check if running in distributed mode.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Community 3` to `Community 1`, `Community 2`, `Community 5`, `Community 6`, `Community 7`, `Community 8`, `Community 9`?**
  _High betweenness centrality (0.272) - this node is a cross-community bridge._
- **Why does `FaultType` connect `Community 1` to `Community 0`, `Community 3`, `Community 4`?**
  _High betweenness centrality (0.173) - this node is a cross-community bridge._
- **Why does `FaceForensicsDataset` connect `Community 5` to `Community 9`, `Community 3`, `Community 6`, `Community 1`?**
  _High betweenness centrality (0.135) - this node is a cross-community bridge._
- **Are the 24 inferred relationships involving `main()` (e.g. with `log_on_main()` and `preextract_frames()`) actually correct?**
  _`main()` has 24 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `CheckpointBitFlipInjector` (e.g. with `Distributed training entry point using torchrun.  This script is designed to b` and `Seed worker for reproducible data loading across epochs.     Ensures each worke`) actually correct?**
  _`CheckpointBitFlipInjector` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `CheckpointManager` (e.g. with `Distributed training entry point using torchrun.  This script is designed to b` and `Seed worker for reproducible data loading across epochs.     Ensures each worke`) actually correct?**
  _`CheckpointManager` has 18 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `main()` (e.g. with `setup_distributed()` and `enforce_determinism()`) actually correct?**
  _`main()` has 15 INFERRED edges - model-reasoned connections that need verification._