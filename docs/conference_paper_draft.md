# Convergence-Aware and Hash-Ring-Based Checkpointing for Fault-Tolerant Distributed Deep Learning

Potta Sai Varshith 
School of Computing  
SASTRA University 
Thanjavur, India
127018044@sastra.ac.in

 
S. Ananthakrishnan 
School of Computing
SASTRA University
Thanjavur, India
ananthakrishnan@cse.sastra.edu

---

## Abstract

Abstract—Checkpointing is a critical fault-tolerance mechanism in distributed deep learning, enabling training jobs to recover from hardware failures without restarting. Conventional strategies such as fixed-epoch snapshotting apply rigid schedules that ignore training convergence dynamics, causing unnecessary I/O overhead. Centralized checkpoint storage further creates recovery bottlenecks in multi-node environments. This paper presents and evaluates a convergence-aware checkpoint scheduler that fits an exponential loss curve online and derives an adaptive checkpoint interval using a closed-form progress-segment formulation. This scheduler is combined with hash-ring-based distributed checkpoint storage to eliminate centralized recovery bottlenecks. We compare four modes: epoch-based, convergence-aware, hash-ring-epoch, and convergence combined with hash ring, using EfficientNet-B0 on FaceForensics across a four-node distributed setup. All modes achieve equivalent model accuracy (73.75% validation). The combined convergence and hash-ring strategy achieves the fastest fault recovery at 36.36 seconds from local cache versus 55.05 seconds for epoch-based central storage recovery, a 34% improvement.

Keywords—distributed deep learning; fault tolerance; checkpointing; convergence-aware scheduling; hash ring; distributed cache;


---

## I. Introduction

Training deep learning models in distributed GPU settings is becoming more prevalent, with extensive models being trained over clusters for durations of hours or even days. In lengthy jobs, hardware failures are anticipated events rather than unusual occurrences. Checkpointing, which involves the regular saving of model states to permanent storage, is the main method for handling failures. In the event of a failure, training starts again from the latest checkpoint, reducing the amount of work lost.

Although it is crucial, the design of checkpointing strategies is still mostly unstructured. The predominant method, epoch-based snapshotting, captures a checkpoint at the conclusion of each training epoch without considering whether the model is in a high-value or low-value convergence state. The Young-Daly formula suggests an ideally optimal timeframe based on mean time between failures (MTBF) but presumes a steady failure rate and overlooks the training loss landscape. Neither method adjusts to the observed convergence behavior.

This paper examines two less explored aspects concurrently: (1) determining when to checkpoint, guided by an online exponential loss fit instead of a predetermined schedule, and (2) deciding where to keep checkpoints, utilizing a hash-ring distributed local cache as opposed to a centralized storage node. We develop a scheduler that takes convergence into account, based on COCI [1], which fits `loss(t) = exp(θ₁·t + θ₂)` over a sliding segment and calculates an adaptive interval from the closed-form progress segment size. We integrate this with a uniform hash-ring storage system and assess all four checkpointing methods in a regulated fault-injection test


---

## II. Background and Related Work

### A. Fixed-Interval and Young-Daly Checkpointing

Fixed-interval checkpointing saves a checkpoint every N epochs or every T seconds. While predictable and easy to implement, it creates unnecessary I/O during stable training phases. The Young-Daly formula gives an optimal interval `T* = √(2 × C × MTBF)`, where C is the checkpoint write time. It assumes a stationary failure distribution and does not account for the varying value of training state across phases [7].

### B. Convergence-Aware Checkpointing

Li et al. [1] introduced COCI, placing checkpoints based on observed loss reduction rather than wall-clock time. Wang et al. [2] proposed Amber for selective incremental checkpointing in LLM training, saving only approximately 2% of significant parameter updates. Gao et al. [3] developed DECK for delta checkpointing in industrial recommendation systems, enabling higher checkpoint frequency on sparse embedding tables. These works address checkpoint timing but not storage distribution.

### C. Distributed Checkpoint Storage

Lee et al. [6] explored hash-ring-based recaching for distributed deep learning caching systems, reducing data redistribution overhead by reassigning only failed-node data. Mahesh et al. [4] proposed ACUTE, a multi-level checkpointing framework for spot VM clusters using memory-based hierarchical checkpoints for faster recovery. Neither system combines convergence-adaptive timing with hash-ring storage.

### D. Research Gap

Existing convergence-aware methods optimize checkpoint timing without addressing distributed storage. Existing hash-ring systems optimize storage without adapting frequency to training dynamics. No prior system unifies both dimensions. This paper fills that gap.

---

## III. System Design

### A. Convergence-Aware Scheduler

The scheduler fits an exponential model `loss(t) = exp(θ₁·t + θ₂)` over a rolling observation segment using online least-squares regression on log-loss values. The decay rate θ₁ is required to be negative; a non-decaying fit triggers no interval update.

The adaptive interval is derived from a closed-form progress-segment formulation. Given the current failure rate λ and checkpoint cost `ts`, the expected progress is:

```
pe = (1 - exp(θ₁ × ts)) × (1/(2λ) - 1/(2θ₁))
```

The scheduler then solves for the elapsed time at which loss decreases by `pe`, yielding the target checkpoint interval. The interval is clamped to `[min_interval, max_interval]`.

To handle non-stationary loss dynamics, the scheduler monitors parameter drift between successive fits. If θ₁ or θ₂ drift beyond configured thresholds, the fitting segment is restarted from the current observation, allowing the scheduler to adapt to new convergence phases such as learning rate decay or loss plateaus.

### B. Hash Ring Storage

Checkpoint ownership is distributed across nodes using a consistent hash ring. Each training node is assigned a position on a virtual ring, and checkpoint files are routed to owning nodes based on their hash position. Checkpoints are stored in local node cache rather than shared central storage. This eliminates the central storage bottleneck and enables local-cache recovery when faults occur, avoiding network round-trips to a remote checkpoint server.

### C. Combined Strategy

The combined convergence and hash-ring strategy applies adaptive interval computation from the convergence-aware scheduler while routing each checkpoint write through the hash-ring storage layer. This aims to reduce both unnecessary checkpoint overhead during stable training phases and recovery latency after failures.

---

## IV. Experimental Setup

All experiments use EfficientNet-B0 trained on FaceForensics for 20 epochs with batch size 2 and world size 4. Four modes are compared: epoch-based, convergence-aware, hash-ring-epoch, and convergence combined with hash ring. The fault model injects a runtime checkpoint fault followed by process restart and resume. Metrics collected include total training time, checkpoint count, checkpoint overhead, validation accuracy (best and final), recovery success, resume source (central storage vs. local cache), resumed epoch, and time to resume.

**Hardware Testbench:** Experiments were conducted on a single physical node equipped with four NVIDIA A16 GPUs (15,356 MiB VRAM each), Driver Version 580.126.09, CUDA Version 13.0. Each GPU was assigned one distributed training process (world size 4), simulating a four-node distributed setup on a shared-memory machine. GPU utilization during idle/checkpoint phases was measured at 0%, with power draw between 12–14W per GPU under the P8 performance state, confirming that checkpoint overhead measurements reflect I/O cost rather than compute cost.

```
+-----------------------------------------------------------------------+
| GPU  Name            | Memory       | Temp | Power       | CUDA  |
|----------------------|--------------|------|-------------|-------|
|  0   NVIDIA A16      | 15356 MiB    | 35°C | 12W / 62W   | 13.0  |
|  1   NVIDIA A16      | 15356 MiB    | 35°C | 13W / 62W   | 13.0  |
|  2   NVIDIA A16      | 15356 MiB    | 32°C | 14W / 62W   | 13.0  |
|  3   NVIDIA A16      | 15356 MiB    | 29°C | 12W / 62W   | 13.0  |
+-----------------------------------------------------------------------+
```

---

## V. Results

### A. Training Phase Metrics

**TABLE I. Training Phase Metrics Across All Four Checkpointing Modes**

| Mode | Train Time (s) | Ckpt Count | Overhead (%) | Best Val Acc | Final Val Acc |
|---|---|---|---|---|---|
| Epoch | 438.08 | 9 | 1.33 | 73.75% | 71.25% |
| Convergence | 436.91 | 10 | 1.41 | 73.75% | 71.25% |
| Hash-Ring Epoch | 464.49 | 9 | 3.47 | 73.75% | 71.25% |
| Conv. + Hash Ring | 477.59 | 23 | 4.41 | 73.75% | 71.25% |

Validation accuracy is identical across all modes (73.75% best, 71.25% final), confirming that checkpointing strategy does not affect model quality. Differences are entirely at the systems level. Epoch mode achieves the lowest checkpoint overhead at 1.33%. The combined convergence and hash-ring mode incurs 4.41% overhead with 23 checkpoints due to the convergence scheduler increasing frequency during rapidly improving loss phases in the short 20-epoch run.

### B. Recovery Phase Metrics

**TABLE II. Fault Recovery Metrics Across All Four Checkpointing Modes**

| Mode | Recovery | Resume Source | From Epoch | Resume Time (s) |
|---|---|---|---|---|
| Epoch | Yes | Central Storage | 10 | 55.05 |
| Convergence | Yes | Central Storage | 17 | 54.06 |
| Hash-Ring Epoch | Yes | Local Cache | 20 | 60.26 |
| Conv. + Hash Ring | Yes | Local Cache | 20 | 36.36 |

The convergence and hash-ring mode resumed from epoch 20 via local cache in 36.36 seconds, 34% faster than epoch mode which resumed from epoch 10 via central storage in 55.05 seconds. Hash-ring epoch also resumed from epoch 20 but took 60.26 seconds, longer than the combined mode. This suggests that convergence-aware timing contributes to faster local-cache recovery by placing checkpoints closer to the fault point, allowing resume from a later epoch with fewer remaining steps.

---

## VI. Discussion

The primary finding is that checkpointing strategy affects recovery behavior and overhead independently of model quality. The identical validation accuracy across all modes establishes a clean comparison context.

Epoch mode is the cheapest strategy during normal training at 1.33% overhead. It is appropriate when storage is abundant and recovery speed is not the primary concern. The convergence-aware mode adds adaptive timing but does not reduce checkpoint count in the current 20-epoch runs, likely because the training run is too short to exhibit significant loss plateaus that would suppress checkpointing intervals.

The hash-ring modes increase training overhead due to distributed cache coordination. However, they enable local-cache recovery that bypasses central storage. The combined mode's 36.36-second recovery is the strongest result of this study. Notably, hash-ring epoch took 60.26 seconds despite local-cache recovery, suggesting that the convergence scheduler's checkpoint placement near the fault point reduces remaining training work after recovery — not just the storage location.

The primary limitation is that evidence is based on a single workload, a small cluster (world size 4), and a limited number of fault injection runs. Generalizing these findings requires evaluation across additional models, datasets, and cluster scales.

---

## VII. Conclusion

This paper presents an implemented comparative study of four checkpointing strategies for fault-tolerant distributed deep learning. A convergence-aware scheduler is designed that fits an exponential loss model online and derives adaptive checkpoint intervals from a closed-form progress-segment formulation. Combined with hash-ring distributed checkpoint storage, this approach achieves the fastest observed recovery time of 36.36 seconds from local cache, a 34% improvement over standard epoch-based central storage recovery at 55.05 seconds, while preserving equivalent model quality across all modes.

Future work includes repeated fault injection campaigns for statistical significance, evaluation on larger clusters and longer training runs where convergence-aware scheduling advantages are more pronounced, and optimization of the hash-ring cache eviction policy to reduce training-phase overhead.

---

## References

[1] Y. Li et al., "Convergence-Aware Optimal Checkpointing (COCI)," Preprint, 2025.

[2] Z. Wang et al., "Amber: Selective Incremental Checkpointing for LLM Training," Preprint, 2025.

[3] X. Gao et al., "DECK: Delta Checkpointing for Industrial Recommendation Systems," Preprint, 2025.

[4] A. Mahesh et al., "ACUTE: Multi-Level Checkpointing for Spot VM Clusters," Preprint, 2025.

[5] Y. Yu et al., "Failure Behavior Analysis in LLM Training on HPC Systems," Preprint, 2025.

[6] J. Lee et al., "Hash Ring-Based Recaching for Distributed Deep Learning," Preprint, 2025.

[7] B. C. Fang et al., "Checkpointing in Distributed Systems," Information Sciences, vol. 496, pp. 300–316, 2019.

[8] B. Joardar et al., "Fault-Tolerant Deep Learning Using Regularization," 2022.
