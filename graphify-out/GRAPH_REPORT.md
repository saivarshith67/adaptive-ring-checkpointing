# Graph Report - D:\Sai\HPC\project\adaptive-ring-checkpointing  (2026-04-30)

## Corpus Check
- 35 files · ~86,694 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 573 nodes · 1234 edges · 86 communities detected
- Extraction: 46% EXTRACTED · 54% INFERRED · 0% AMBIGUOUS · INFERRED: 665 edges (avg confidence: 0.58)
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
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 79|Community 79]]
- [[_COMMUNITY_Community 80|Community 80]]
- [[_COMMUNITY_Community 81|Community 81]]
- [[_COMMUNITY_Community 82|Community 82]]
- [[_COMMUNITY_Community 83|Community 83]]
- [[_COMMUNITY_Community 84|Community 84]]
- [[_COMMUNITY_Community 85|Community 85]]

## God Nodes (most connected - your core abstractions)
1. `CheckpointBitFlipInjector` - 68 edges
2. `MetricsCollector` - 66 edges
3. `CheckpointManager` - 64 edges
4. `FaultType` - 59 edges
5. `CheckpointFaultConfig` - 57 edges
6. `main()` - 56 edges
7. `HashRingCheckpointManager` - 56 edges
8. `FaceForensicsDataset` - 55 edges
9. `VideoFaceForensicsDataset` - 52 edges
10. `ConvergenceAwareScheduler` - 50 edges

## Surprising Connections (you probably didn't know these)
- `CheckpointManager` --uses--> `Checkpoint managers for framework-native checkpointing backends.`  [INFERRED]
  D:\Sai\HPC\project\adaptive-ring-checkpointing\src\coci\checkpointing\checkpoint_manager.py → D:\Sai\HPC\project\adaptive-ring-checkpointing\src\coci\checkpointing\framework_checkpoint_manager.py
- `CheckpointManager` --uses--> `Checkpoint manager that delegates saves to framework-native APIs when possible.`  [INFERRED]
  D:\Sai\HPC\project\adaptive-ring-checkpointing\src\coci\checkpointing\checkpoint_manager.py → D:\Sai\HPC\project\adaptive-ring-checkpointing\src\coci\checkpointing\framework_checkpoint_manager.py
- `CheckpointManager` --uses--> `Create the right checkpoint manager for a backend name.`  [INFERRED]
  D:\Sai\HPC\project\adaptive-ring-checkpointing\src\coci\checkpointing\checkpoint_manager.py → D:\Sai\HPC\project\adaptive-ring-checkpointing\src\coci\checkpointing\framework_checkpoint_manager.py
- `CheckpointManager` --uses--> `Checkpoint manager that layers hash-ring local shard caching over normal checkpo`  [INFERRED]
  D:\Sai\HPC\project\adaptive-ring-checkpointing\src\coci\checkpointing\checkpoint_manager.py → D:\Sai\HPC\project\adaptive-ring-checkpointing\src\coci\checkpointing\hash_ring_checkpoint_manager.py
- `CheckpointManager` --uses--> `Start the heartbeat fault-detection thread for liveness monitoring.          A`  [INFERRED]
  D:\Sai\HPC\project\adaptive-ring-checkpointing\src\coci\checkpointing\checkpoint_manager.py → D:\Sai\HPC\project\adaptive-ring-checkpointing\src\coci\checkpointing\hash_ring_checkpoint_manager.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.03
Nodes (81): get_cifar100_dataset(), ConvergenceMetrics, DistributedMetrics, EpochMetrics, ExperimentSummary, FaultMetrics, FrameworkCheckpointMetrics, HashRingMetrics (+73 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (87): CheckpointBitFlipInjector, CheckpointFaultConfig, FaultType, Inject bit-flips into checkpoint tensors before model/optimizer restore., CheckpointManager, Return checkpoint-manager runtime metrics in a hash-ring-compatible shape., ConvergenceAwareScheduler, ConvergenceFit (+79 more)

### Community 2 - "Community 2"
Cohesion: 0.05
Nodes (20): enforce_determinism(), Force deterministic behavior to isolate bit-flip impact from training noise., Enum, FaultDetector, NodeStatus, Broadcast heartbeat signals to all peers using dist.all_reduce.          Returns, Gather suspicion votes from all ranks using dist.all_gather.          Returns ag, Start background thread for periodic heartbeat broadcast.          Args: (+12 more)

### Community 3 - "Community 3"
Cohesion: 0.05
Nodes (43): Save a checkpoint.          Args:             model: The model to save (or DD, Record where a checkpoint was loaded from., Record detected fault., Record that a resume was attempted., Record a successful resume event and its rollback characteristics., Record the last observed failure state., Record recovery attempt.                  Args:             success: Whether, Record the latest known training progress for recovery metrics. (+35 more)

### Community 4 - "Community 4"
Cohesion: 0.09
Nodes (11): ElasticRecaching, HashRingCheckpointManager, Checkpoint manager that layers hash-ring local shard caching over normal checkpo, Start the heartbeat fault-detection thread for liveness monitoring.          A, Stop the heartbeat fault-detection thread gracefully.          Args:, Return runtime metrics for resilience analysis and reporting., Write shard to local NVMe cache with atomic rename pattern., Load shard from local cache with verification. (+3 more)

### Community 5 - "Community 5"
Cohesion: 0.14
Nodes (6): CheckpointManager, create_checkpoint_manager(), FrameworkCheckpointManager, Checkpoint managers for framework-native checkpointing backends., Checkpoint manager that delegates saves to framework-native APIs when possible., Create the right checkpoint manager for a backend name.

### Community 6 - "Community 6"
Cohesion: 0.12
Nodes (11): Load the latest checkpoint and restore model/optimizer state.          Args:, export_csv_table(), export_markdown_table(), generate_full_run_rows(), generate_recovery_rows(), load_rows(), main(), save_bar_plot() (+3 more)

### Community 7 - "Community 7"
Cohesion: 0.2
Nodes (9): ABC, CheckpointStrategy, CheckpointStrategyFactory, create(), EpochStrategy, FixedIntervalStrategy, FrameworkEpochStrategy, Epoch checkpoint trigger for framework-native checkpoint backends. (+1 more)

### Community 8 - "Community 8"
Cohesion: 0.24
Nodes (8): get_efficientnet_binary(), get_model(), get_multiframe_model(), MultiFrameModel, Multi-frame video classification model.      Processes multiple frames through, Forward pass.          Args:             x: Input tensor of shape (B, T, C, H, Create EfficientNet-B0 model for binary classification.      Args:         pr, Create a multi-frame model for video-level deepfake detection.      This model

### Community 9 - "Community 9"
Cohesion: 0.22
Nodes (5): get_faceforensics_transforms(), Load samples from pre-computed crop cache., Set up for on-the-fly MTCNN detection (slow)., Get transforms for FaceForensics++ face images.      Args:         include_re, Load all .npy file paths from real/fake directories.

### Community 10 - "Community 10"
Cohesion: 0.33
Nodes (2): Dataset, ImageDataset

### Community 11 - "Community 11"
Cohesion: 0.33
Nodes (3): Load all image paths from the dataset directory., Recursively collect image paths from video folder structure., Load images from flat directory structure.

### Community 12 - "Community 12"
Cohesion: 0.33
Nodes (3): Get a single sample from the dataset.          If using pre-computed crops (re, Load pre-computed face crop from numpy file., Detect face using MTCNN (slow).

### Community 13 - "Community 13"
Cohesion: 0.33
Nodes (2): FaultInjector, Rank-aware fault injector for simulating GPU/node failures.      Supports two

### Community 14 - "Community 14"
Cohesion: 0.33
Nodes (6): Adaptive Ring Checkpointing, Convergence-Aware Checkpointing (COCI), Soft Error Injection Mechanism, Hash Ring Checkpointing, Loss Curve Model (Exponential Decay), Piecewise Online Fitting (POF)

### Community 15 - "Community 15"
Cohesion: 0.5
Nodes (2): Get frames from a single video.          Args:             idx: Index of the, Load a single frame and apply transforms.

### Community 16 - "Community 16"
Cohesion: 0.5
Nodes (4): End-to-End Fault Tolerance Verification, Phase 4 - Exception Handling, Phase 5 - Fault Injection, Phase 6 - Checkpoint Recovery

### Community 17 - "Community 17"
Cohesion: 1.0
Nodes (2): Config, load_config()

### Community 18 - "Community 18"
Cohesion: 1.0
Nodes (0): 

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (1): Get frames from a single video.          Returns:             tuple: (frames_

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
Nodes (0): 

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (0): 

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (0): 

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (0): 

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (0): 

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): Check if running in distributed mode.

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): Distributed training metrics.

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): Overall experiment summary.

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): Collects metrics throughout training for later export and analysis.

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (1): Initialize metrics collector.                  Args:             experiment_n

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): Mark the start of an epoch.

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): Record metrics for completed epoch.                  Args:             epoch:

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (1): Record checkpoint metrics.                  Args:             checkpoint_size

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): Record in-epoch checkpoint metrics (e.g., convergence-based).

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): Update convergence-aware scheduler metrics.                  Args:

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Increment convergence-based checkpoint counter.

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Record that convergence fitting segment restarted (POF).

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): Update hash-ring checkpoint metrics.                  Args:             total

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): Record local cache hit.

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): Record where a checkpoint was loaded from.

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): Record a shard being written to local cache.

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): Record shard reassignment after a failure.

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Record shard recovery event.                  Args:             recovery_time

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): Record injected fault.

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): Record a runtime fault injection event.

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Record that a resume was attempted.

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Record a successful resume event and its rollback characteristics.

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Record the last observed failure state.

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): Record recovery attempt.                  Args:             success: Whether r

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): Record checkpoint integrity check failure.

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): Set distributed training configuration.                  Args:             wo

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): Record collective synchronization operation.                  Args:

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): Increment batch counter.                  Args:             samples: Number of s

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): Record the latest known training progress for recovery metrics.

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (1): Get all epoch metrics as dictionaries for export.

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (1): Get current state snapshot for debugging.

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (1): Save a checkpoint.          Args:             model: The model to save (or DD

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (1): Load the latest checkpoint and restore model/optimizer state.          Args:

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (1): Fault injection and recovery metrics.

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (1): Distributed training metrics.

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (1): Overall experiment summary.

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (1): Collects metrics throughout training for later export and analysis.

### Community 66 - "Community 66"
Cohesion: 1.0
Nodes (1): Initialize metrics collector.                  Args:             experiment_n

### Community 67 - "Community 67"
Cohesion: 1.0
Nodes (1): Mark the start of an epoch.

### Community 68 - "Community 68"
Cohesion: 1.0
Nodes (1): Record metrics for completed epoch.                  Args:             epoch:

### Community 69 - "Community 69"
Cohesion: 1.0
Nodes (1): Record checkpoint metrics.                  Args:             checkpoint_size

### Community 70 - "Community 70"
Cohesion: 1.0
Nodes (1): Record in-epoch checkpoint metrics (e.g., convergence-based).

### Community 71 - "Community 71"
Cohesion: 1.0
Nodes (1): Increment convergence-based checkpoint counter.

### Community 72 - "Community 72"
Cohesion: 1.0
Nodes (1): Record that convergence fitting segment restarted (POF).

### Community 73 - "Community 73"
Cohesion: 1.0
Nodes (1): Update hash-ring checkpoint metrics.                  Args:             total

### Community 74 - "Community 74"
Cohesion: 1.0
Nodes (1): Record local cache hit.

### Community 75 - "Community 75"
Cohesion: 1.0
Nodes (1): Record local cache miss.

### Community 76 - "Community 76"
Cohesion: 1.0
Nodes (1): Record shard recovery event.                  Args:             recovery_time

### Community 77 - "Community 77"
Cohesion: 1.0
Nodes (1): Record injected fault.

### Community 78 - "Community 78"
Cohesion: 1.0
Nodes (1): Record detected fault.

### Community 79 - "Community 79"
Cohesion: 1.0
Nodes (1): Record recovery attempt.                  Args:             success: Whether

### Community 80 - "Community 80"
Cohesion: 1.0
Nodes (1): Set distributed training configuration.                  Args:             wo

### Community 81 - "Community 81"
Cohesion: 1.0
Nodes (1): Record collective synchronization operation.                  Args:

### Community 82 - "Community 82"
Cohesion: 1.0
Nodes (1): Increment batch counter.                  Args:             samples: Number o

### Community 83 - "Community 83"
Cohesion: 1.0
Nodes (1): Generate final summary metrics.

### Community 84 - "Community 84"
Cohesion: 1.0
Nodes (1): Get all epoch metrics as dictionaries for export.

### Community 85 - "Community 85"
Cohesion: 1.0
Nodes (1): Get current state snapshot for debugging.

## Knowledge Gaps
- **163 isolated node(s):** `Example of training loop with metrics collection.`, `Example of comparing results across modes.`, `Distributed training utilities for multi-GPU coordination.  Provides functions`, `Check if the default process group is initialized.      Returns:         bool`, `Get the rank of the current process in the distributed group.      Returns:` (+158 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 18`** (2 nodes): `main()`, `main.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (2 nodes): `Get frames from a single video.          Returns:             tuple: (frames_`, `.__getitem__()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `train_convergence_hashring.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (1 nodes): `train_convergence_normal.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `train_deepspeed.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `train_epoch_based.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `train_fsdp.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `train_hashring_epoch.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `train_hf_trainer.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `train_pytorch_lightning.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `train_wandb_artifacts.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `Check if running in distributed mode.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `Distributed training metrics.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Overall experiment summary.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `Collects metrics throughout training for later export and analysis.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `Initialize metrics collector.                  Args:             experiment_n`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `Mark the start of an epoch.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `Record metrics for completed epoch.                  Args:             epoch:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `Record checkpoint metrics.                  Args:             checkpoint_size`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `Record in-epoch checkpoint metrics (e.g., convergence-based).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Update convergence-aware scheduler metrics.                  Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Increment convergence-based checkpoint counter.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Record that convergence fitting segment restarted (POF).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Update hash-ring checkpoint metrics.                  Args:             total`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Record local cache hit.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Record where a checkpoint was loaded from.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `Record a shard being written to local cache.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `Record shard reassignment after a failure.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Record shard recovery event.                  Args:             recovery_time`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `Record injected fault.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `Record a runtime fault injection event.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Record that a resume was attempted.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Record a successful resume event and its rollback characteristics.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Record the last observed failure state.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Record recovery attempt.                  Args:             success: Whether r`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Record checkpoint integrity check failure.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `Set distributed training configuration.                  Args:             wo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `Record collective synchronization operation.                  Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `Increment batch counter.                  Args:             samples: Number of s`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `Record the latest known training progress for recovery metrics.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `Get all epoch metrics as dictionaries for export.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (1 nodes): `Get current state snapshot for debugging.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (1 nodes): `Save a checkpoint.          Args:             model: The model to save (or DD`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (1 nodes): `Load the latest checkpoint and restore model/optimizer state.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (1 nodes): `Fault injection and recovery metrics.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (1 nodes): `Distributed training metrics.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (1 nodes): `Overall experiment summary.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (1 nodes): `Collects metrics throughout training for later export and analysis.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 66`** (1 nodes): `Initialize metrics collector.                  Args:             experiment_n`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 67`** (1 nodes): `Mark the start of an epoch.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 68`** (1 nodes): `Record metrics for completed epoch.                  Args:             epoch:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 69`** (1 nodes): `Record checkpoint metrics.                  Args:             checkpoint_size`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 70`** (1 nodes): `Record in-epoch checkpoint metrics (e.g., convergence-based).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 71`** (1 nodes): `Increment convergence-based checkpoint counter.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 72`** (1 nodes): `Record that convergence fitting segment restarted (POF).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 73`** (1 nodes): `Update hash-ring checkpoint metrics.                  Args:             total`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 74`** (1 nodes): `Record local cache hit.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 75`** (1 nodes): `Record local cache miss.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 76`** (1 nodes): `Record shard recovery event.                  Args:             recovery_time`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 77`** (1 nodes): `Record injected fault.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 78`** (1 nodes): `Record detected fault.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 79`** (1 nodes): `Record recovery attempt.                  Args:             success: Whether`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 80`** (1 nodes): `Set distributed training configuration.                  Args:             wo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 81`** (1 nodes): `Record collective synchronization operation.                  Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 82`** (1 nodes): `Increment batch counter.                  Args:             samples: Number o`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 83`** (1 nodes): `Generate final summary metrics.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 84`** (1 nodes): `Get all epoch metrics as dictionaries for export.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 85`** (1 nodes): `Get current state snapshot for debugging.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Community 3` to `Community 0`, `Community 1`, `Community 2`, `Community 4`, `Community 5`, `Community 6`, `Community 8`, `Community 9`?**
  _High betweenness centrality (0.341) - this node is a cross-community bridge._
- **Why does `MetricsCollector` connect `Community 0` to `Community 1`, `Community 3`?**
  _High betweenness centrality (0.201) - this node is a cross-community bridge._
- **Why does `CheckpointManager` connect `Community 1` to `Community 0`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 6`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Are the 56 inferred relationships involving `CheckpointBitFlipInjector` (e.g. with `Distributed training entry point using torchrun.  This script is designed to b` and `Seed worker for reproducible data loading across epochs.     Ensures each worke`) actually correct?**
  _`CheckpointBitFlipInjector` has 56 INFERRED edges - model-reasoned connections that need verification._
- **Are the 32 inferred relationships involving `MetricsCollector` (e.g. with `MetricsExporter` and `Export metrics to various formats for analysis and plotting.`) actually correct?**
  _`MetricsCollector` has 32 INFERRED edges - model-reasoned connections that need verification._
- **Are the 58 inferred relationships involving `CheckpointManager` (e.g. with `Distributed training entry point using torchrun.  This script is designed to b` and `Seed worker for reproducible data loading across epochs.     Ensures each worke`) actually correct?**
  _`CheckpointManager` has 58 INFERRED edges - model-reasoned connections that need verification._
- **Are the 56 inferred relationships involving `FaultType` (e.g. with `Distributed training entry point using torchrun.  This script is designed to b` and `Seed worker for reproducible data loading across epochs.     Ensures each worke`) actually correct?**
  _`FaultType` has 56 INFERRED edges - model-reasoned connections that need verification._