# Graph Report - D:\Sai\HPC\project\adaptive-ring-checkpointing  (2026-04-18)

## Corpus Check
- 24 files · ~23,801 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 320 nodes · 651 edges · 22 communities detected
- Extraction: 56% EXTRACTED · 44% INFERRED · 0% AMBIGUOUS · INFERRED: 284 edges (avg confidence: 0.61)
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

## God Nodes (most connected - your core abstractions)
1. `CheckpointManager` - 44 edges
2. `FaceForensicsDataset` - 35 edges
3. `main()` - 32 edges
4. `VideoFaceForensicsDataset` - 32 edges
5. `HashRingCheckpointManager` - 28 edges
6. `VideoDatasetFast` - 28 edges
7. `CheckpointBitFlipInjector` - 27 edges
8. `main()` - 23 edges
9. `FaultInjector` - 22 edges
10. `ConvergenceAwareScheduler` - 19 edges

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
Nodes (48): CheckpointManager, ConvergenceAwareScheduler, ConvergenceFit, COCI-inspired online scheduler for convergence-aware checkpoint timing.      T, FaceForensicsDataset, get_faceforensics_transforms(), PyTorch Dataset for FaceForensics++ deepfake detection.      Supports two mode, Load samples from pre-computed crop cache. (+40 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (18): Enum, FaultDetector, NodeStatus, Broadcast heartbeat signals to all peers using dist.all_reduce.          Returns, Gather suspicion votes from all ranks using dist.all_gather.          Returns ag, Start background thread for periodic heartbeat broadcast.          Args:, Stop the heartbeat thread gracefully.          Args:             timeout: Maximu, Background loop that periodically broadcasts heartbeats. (+10 more)

### Community 2 - "Community 2"
Cohesion: 0.08
Nodes (35): Save a checkpoint.          Args:             model: The model to save (or DD, barrier(), cleanup_distributed(), get_local_rank(), get_rank(), get_world_size(), is_distributed_initialized(), is_main_process() (+27 more)

### Community 3 - "Community 3"
Cohesion: 0.13
Nodes (24): CheckpointBitFlipInjector, CheckpointFaultConfig, enforce_determinism(), FaultType, Force deterministic behavior to isolate bit-flip impact from training noise., Inject bit-flips into checkpoint tensors before model/optimizer restore., Load the latest checkpoint and restore model/optimizer state.          Args:, str (+16 more)

### Community 4 - "Community 4"
Cohesion: 0.09
Nodes (9): ElasticRecaching, HashRingCheckpointManager, Checkpoint manager that layers hash-ring local shard caching over normal checkpo, Write shard to local NVMe cache with atomic rename pattern., Load shard from local cache with verification., Verify shard integrity by loading and checking., Return cache usage statistics., Remove cached files for shards owned by a dead node. (+1 more)

### Community 5 - "Community 5"
Cohesion: 0.11
Nodes (13): ABC, get_cifar100_dataset(), Config, load_config(), CheckpointStrategy, CheckpointStrategyFactory, create(), EpochStrategy (+5 more)

### Community 6 - "Community 6"
Cohesion: 0.24
Nodes (8): get_efficientnet_binary(), get_model(), get_multiframe_model(), MultiFrameModel, Multi-frame video classification model.      Processes multiple frames through, Forward pass.          Args:             x: Input tensor of shape (B, T, C, H, Create EfficientNet-B0 model for binary classification.      Args:         pr, Create a multi-frame model for video-level deepfake detection.      This model

### Community 7 - "Community 7"
Cohesion: 0.22
Nodes (9): Adaptive Ring Checkpointing, graphify, Hash Ring Algorithm, Horovod, HPC Cluster, Lee et al. - Fault-Tolerant Deep Learning Cache with Hash Ring, PyTorch, SC '24 Workshop (+1 more)

### Community 8 - "Community 8"
Cohesion: 0.33
Nodes (2): Dataset, ImageDataset

### Community 9 - "Community 9"
Cohesion: 0.33
Nodes (3): Load all image paths from the dataset directory., Recursively collect image paths from video folder structure., Load images from flat directory structure.

### Community 10 - "Community 10"
Cohesion: 0.33
Nodes (3): Get a single sample from the dataset.          If using pre-computed crops (re, Load pre-computed face crop from numpy file., Detect face using MTCNN (slow).

### Community 11 - "Community 11"
Cohesion: 0.4
Nodes (5): Phase 4 - Exception Handling, Phase 5 - Fault Injection, Phase 6 - Checkpoint Recovery, End-to-End Fault Tolerance Verification, Fault Tolerance Verification Results

### Community 12 - "Community 12"
Cohesion: 0.5
Nodes (2): Get frames from a single video.          Args:             idx: Index of the, Load a single frame and apply transforms.

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
Nodes (0): 

### Community 18 - "Community 18"
Cohesion: 1.0
Nodes (1): Check if running in distributed mode.

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (1): Save a checkpoint.          Args:             model: The model to save (or DD

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (1): Load the latest checkpoint and restore model/optimizer state.          Args:

### Community 21 - "Community 21"
Cohesion: 1.0
Nodes (1): Epoch Rollback Protocol

## Knowledge Gaps
- **68 isolated node(s):** `Distributed training utilities for multi-GPU coordination.  Provides functions`, `Check if the default process group is initialized.      Returns:         bool`, `Get the rank of the current process in the distributed group.      Returns:`, `Get the total number of processes in the distributed group.      Returns:`, `Get the local rank of the current process.      The local rank is the GPU devi` (+63 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 13`** (2 nodes): `main()`, `main.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 14`** (1 nodes): `train_convergence_hashring.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 15`** (1 nodes): `train_convergence_normal.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 16`** (1 nodes): `train_epoch_based.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 17`** (1 nodes): `train_hashring_epoch.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 18`** (1 nodes): `Check if running in distributed mode.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (1 nodes): `Save a checkpoint.          Args:             model: The model to save (or DD`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `Load the latest checkpoint and restore model/optimizer state.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (1 nodes): `Epoch Rollback Protocol`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Community 2` to `Community 0`, `Community 3`, `Community 4`, `Community 6`?**
  _High betweenness centrality (0.238) - this node is a cross-community bridge._
- **Why does `FaultType` connect `Community 3` to `Community 0`, `Community 1`, `Community 2`, `Community 5`?**
  _High betweenness centrality (0.147) - this node is a cross-community bridge._
- **Why does `CheckpointManager` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 5`?**
  _High betweenness centrality (0.135) - this node is a cross-community bridge._
- **Are the 39 inferred relationships involving `CheckpointManager` (e.g. with `Distributed training entry point using torchrun.  This script is designed to b` and `Seed worker for reproducible data loading across epochs.     Ensures each worke`) actually correct?**
  _`CheckpointManager` has 39 INFERRED edges - model-reasoned connections that need verification._
- **Are the 22 inferred relationships involving `FaceForensicsDataset` (e.g. with `Seed worker for reproducible data loading.` and `Create DataLoader with DistributedSampler for FaceForensics++ dataset.      Ar`) actually correct?**
  _`FaceForensicsDataset` has 22 INFERRED edges - model-reasoned connections that need verification._
- **Are the 27 inferred relationships involving `main()` (e.g. with `log_on_main()` and `preextract_frames()`) actually correct?**
  _`main()` has 27 INFERRED edges - model-reasoned connections that need verification._
- **Are the 22 inferred relationships involving `VideoFaceForensicsDataset` (e.g. with `Seed worker for reproducible data loading.` and `Create DataLoader with DistributedSampler for FaceForensics++ dataset.      Ar`) actually correct?**
  _`VideoFaceForensicsDataset` has 22 INFERRED edges - model-reasoned connections that need verification._