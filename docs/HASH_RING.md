# Hash Ring Fault-Tolerant Checkpoint System — Implementation Spec Sheet

**Derived from:** *Fault-Tolerant Deep Learning Cache with Hash Ring for Load Balancing in HPC Systems* (Lee et al., SC '24 Workshop, 2024)  
**Target Environment:** HPC Cluster with Central Storage Server + GPU Nodes  
**Version:** 1.0  
**Status:** Draft

---

## 1. Problem Statement

Standard HPC GPU clusters with a central checkpoint storage server face two compounding failure risks:

1. **Node failures mid-training** — a GPU node dies after hours of training, and the job either crashes or falls back to slow central storage access for every subsequent I/O, causing a straggler cascade.
2. **No recovery scheduling** — upon restart, all GPU nodes blindly re-read from the central storage with no coordination on who reads what, and from which epoch checkpoint, creating I/O bottlenecks and uneven resumption.

The paper's core contribution — a **hash ring-based elastic recaching** system on HVAC — is adapted here for a cluster where:
- Checkpoints are stored on a **central storage server** (analogous to the Parallel File System, PFS, in the paper).
- GPU nodes access checkpoints over the network.
- On node failure, surviving GPUs must **redistribute checkpoint responsibility** and **resume training from a well-defined, coordinated point**.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING CLUSTER                         │
│                                                                 │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   │
│   │  GPU-0   │   │  GPU-1   │   │  GPU-2   │   │  GPU-N   │   │
│   │ NVMe SSDs│   │ NVMe SSDs│   │ NVMe SSDs│   │ NVMe SSDs│   │
│   │ (local   │   │ (local   │   │ (local   │   │ (local   │   │
│   │  cache)  │   │  cache)  │   │  cache)  │   │  cache)  │   │
│   └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘   │
│        └──────────────┴──────────────┴───────────────┘         │
│                              │                                  │
│                   ┌──────────▼──────────┐                       │
│                   │   Hash Ring Manager  │                       │
│                   │  (coordinator node)  │                       │
│                   └──────────┬──────────┘                       │
│                              │                                  │
└──────────────────────────────┼──────────────────────────────────┘
                               │ Network (InfiniBand / Ethernet)
                               │
              ┌────────────────▼────────────────┐
              │       Central Storage Server     │
              │  (Checkpoints + Training Data)   │
              │    [Ground truth / PFS analog]   │
              └─────────────────────────────────┘
```

**Key design principle:** The central storage is the authoritative store. GPU-local NVMe acts as a fast cache layer. The hash ring manages which GPU owns which checkpoint shards. After a failure, the hash ring re-routes ownership with **one PFS access per lost shard**, not one per epoch.

---

## 3. Core Components

### 3.1 Hash Ring Manager

The hash ring manager is a lightweight coordinator (can run as a sidecar process on the head/coordinator node or as a daemon on each GPU node with a gossip-style consensus).

**Responsibilities:**
- Maintains the circular hash ring mapping checkpoint shards → GPU nodes.
- Tracks liveness of all GPU nodes via heartbeat timeouts.
- On node failure: removes the dead node's virtual nodes from the ring and recomputes shard ownership.
- Broadcasts updated ring state to all surviving nodes.

**Data structures:**
```
ring: std::map<float, NodeID>          // sorted map of hash_value → node
virtual_nodes: map<NodeID, [float]>    // physical node → list of virtual positions
shard_table: map<ShardID, NodeID>      // checkpoint shard → current owner node
node_status: map<NodeID, Status>       // ALIVE | SUSPECTED | DEAD
```

**Hash function:**
```
position(key) = SHA256(key) mod 1  // normalized [0.0, 1.0) range
node_position(node_id, vnode_idx) = SHA256(node_id + ":" + vnode_idx) mod 1
```

**Virtual node count:** Set to **100 virtual nodes per physical GPU node** (paper's optimal value for their environment; tune based on your shard-to-node ratio).

---

### 3.2 Checkpoint Shard Manager

Checkpoints are split into fixed-size shards. Each shard is assigned to a GPU node via the hash ring. The GPU node caches its assigned shards on local NVMe.

**Shard assignment:**
```
shard_owner(shard_id) = ring.upper_bound(position(shard_id)) → nearest clockwise node
```

**Shard lifecycle:**
```
UNCACHED   → node requests shard from central storage → CACHING → CACHED
CACHED     → node failure detected                    → ORPHANED
ORPHANED   → new owner re-fetches from central storage → RE-CACHED
```

**Shard metadata record:**
```json
{
  "shard_id": "ckpt_epoch3_layer7_shard42",
  "epoch": 3,
  "layer": 7,
  "size_bytes": 536870912,
  "owner_node": "gpu-node-04",
  "status": "CACHED",
  "central_path": "/storage/checkpoints/epoch3/layer7/shard42.pt",
  "cached_path": "/nvme/cache/shard42.pt",
  "last_verified": "2025-11-17T14:23:00Z"
}
```

---

### 3.3 Fault Detector

Each GPU node runs a local fault detector daemon. Detection is timeout-based to avoid false positives from transient network spikes.

**Algorithm:**
```
for each RPC request to node X:
    if response_time > TTL:
        increment timeout_counter[X]
    if timeout_counter[X] >= TIMEOUT_LIMIT:
        mark node X as SUSPECTED
        broadcast SUSPECTED(X) to all peers
        if majority acknowledge:
            mark node X as DEAD
            trigger hash ring update
```

**Tuning parameters:**

| Parameter | Default | Notes |
|---|---|---|
| `TTL` (Time-to-Live) | 5 seconds | Must exceed max observed network latency |
| `TIMEOUT_LIMIT` | 3 | Number of consecutive timeouts before declaring failure |
| `HEARTBEAT_INTERVAL` | 1 second | Frequency of liveness pings |
| `SUSPICION_QUORUM` | 51% of nodes | Majority needed to confirm failure |

---

### 3.4 Elastic Recaching Engine

On failure confirmation, the recaching engine handles re-routing and one-time re-population from central storage.

**Recaching flow (post-failure):**

```
1. Hash ring removes dead node's virtual positions
2. For each orphaned shard S:
     new_owner = ring.upper_bound(position(S)) → next clockwise node
     shard_table[S] = new_owner
3. Training I/O requests for S are redirected to new_owner
4. new_owner checks local NVMe:
     if CACHED: serve directly
     else (first access after failure):
         fetch from central storage  ← ONE-TIME PFS ACCESS
         serve to requester
         cache to local NVMe        ← subsequent accesses hit NVMe
```

**Critical property:** Each lost shard triggers exactly **one** central storage access across all surviving epochs. Contrast with PFS-redirection (naive fallback), which accesses central storage on *every* epoch for every lost shard.

---

## 4. Recovery Scheduling

This section addresses the cluster-specific requirement: **coordinated GPU restart scheduling** after one or more node failures, so GPUs resume from the correct checkpoint with minimal redundant work.

### 4.1 Epoch Rollback Protocol

Following the paper's use of Horovod Elastic Run, the recovery scheduler must roll all surviving GPUs back to the **start of the most recently completed epoch** before the failure. This is the "victim epoch" — the epoch during which the failure was detected.

```
last_clean_epoch = max epoch for which ALL shards are CACHED or RE-CACHED
victim_epoch     = last_clean_epoch + 1   (the epoch in progress during failure)
resume_from      = last_clean_epoch       (roll back to start of this epoch)
```

**Epoch checkpoint record (stored on central storage):**
```json
{
  "epoch": 3,
  "status": "COMPLETE",
  "model_state_path": "/storage/checkpoints/epoch3/model.pt",
  "optimizer_state_path": "/storage/checkpoints/epoch3/optimizer.pt",
  "shard_manifest": "/storage/checkpoints/epoch3/manifest.json",
  "participants": ["gpu-node-00", "gpu-node-01", ..., "gpu-node-N"],
  "completed_at": "2025-11-17T14:00:00Z"
}
```

### 4.2 GPU Resume Assignment

After a node failure, the surviving N−k nodes must be assigned work slices. The scheduler divides the dataset shards among surviving nodes so that no node is idle and the load is balanced.

**Resume assignment algorithm:**

```
Input:  surviving_nodes  (list of N−k active GPU nodes)
        total_shards     (total number of data shards for this epoch)
        shard_table      (current hash ring shard → node mapping)

Step 1: Re-balance hash ring
        For each surviving node n:
            assign virtual_nodes[n] positions on ring

Step 2: Re-map shards to surviving nodes
        For each shard S in total_shards:
            shard_table[S] = ring.upper_bound(position(S))

Step 3: Compute per-node work slice
        For each surviving node n:
            work_slice[n] = [S for S in total_shards if shard_table[S] == n]

Step 4: Broadcast assignments
        Send work_slice[n] to each GPU node n
        Include: resume_epoch, checkpoint_path, shard list

Step 5: Trigger synchronized resume
        All nodes load model state from checkpoint epoch = resume_from
        Each node processes only its assigned work_slice
        Synchronize via barrier at end of each mini-batch
```

**Assignment scheduling table (example: 8 GPUs, 1 failure → 7 survivors):**

| GPU Node | Assigned Shards (Before Failure) | Reassigned Shards (After Failure) | Δ Load |
|---|---|---|---|
| gpu-node-00 | 128 shards | 147 shards | +19 |
| gpu-node-01 | 128 shards | 146 shards | +18 |
| gpu-node-02 | 128 shards | 146 shards | +18 |
| gpu-node-03 | **DEAD** | — | — |
| gpu-node-04 | 128 shards | 147 shards | +19 |
| gpu-node-05 | 128 shards | 146 shards | +18 |
| gpu-node-06 | 128 shards | 147 shards | +19 |
| gpu-node-07 | 128 shards | 147 shards | +19 |

With 100 virtual nodes per physical node, the extra shards from the dead node are spread across **all surviving nodes** (not just the one adjacent in the ring), achieving near-uniform distribution.

### 4.3 Checkpoint Resume Modes

The scheduler supports three resume modes depending on failure timing:

| Mode | Trigger Condition | Action |
|---|---|---|
| **Hot Resume** | Failure in epoch E, >80% of epoch complete | Resume from mid-epoch using micro-checkpoints if available; else roll back to epoch E start |
| **Cold Resume** | Failure in epoch E, <80% complete | Roll back to epoch E−1 checkpoint; re-run epoch E from scratch with N−k nodes |
| **Critical Failure** | Multiple simultaneous failures (>20% nodes lost) | Halt, alert operator, wait for node re-provisioning before restarting |

**Mode selection logic:**
```python
def select_resume_mode(epoch_progress, nodes_lost, total_nodes):
    loss_ratio = nodes_lost / total_nodes
    if loss_ratio > 0.20:
        return "CRITICAL_FAILURE"
    elif epoch_progress >= 0.80:
        return "HOT_RESUME"
    else:
        return "COLD_RESUME"
```

---

## 5. Central Storage Integration

### 5.1 Checkpoint Write Protocol

Checkpoints are written to central storage at the **end of every completed epoch** and at configurable **micro-checkpoint intervals** within an epoch.

**Write flow:**
```
Training epoch completes
    → Each GPU node flushes its local model shard to central storage
    → Central storage assembles and validates the full checkpoint
    → Central storage writes epoch manifest
    → Hash Ring Manager marks epoch as COMMITTED
    → Old epoch checkpoint retained until epoch+2 is COMMITTED (rolling window)
```

**Storage path convention:**
```
/storage/checkpoints/
    epoch_{N}/
        model.pt               ← full model state
        optimizer.pt           ← optimizer state
        manifest.json          ← shard map + participant list + epoch metadata
        shards/
            shard_{id}.pt      ← individual tensor shards (optional, for large models)
    micro/
        epoch_{N}_step_{M}.pt  ← micro-checkpoints within epoch
```

### 5.2 Read Access Pattern

Under normal operation, GPU nodes never read from central storage after the first epoch — all reads are served from local NVMe. Central storage reads occur only in two cases:

1. **First epoch (cold start):** All shards fetched once, cached on NVMe.
2. **Post-failure recaching:** Orphaned shards fetched once by new owners.

This mirrors the paper's result: FT w/NVMe incurred only a **12.5–26.7% runtime increase** versus **32.2–68.7%** for naive PFS redirection.

---

## 6. Implementation Checklist

### Phase 1 — Core Infrastructure

- [ ] Implement hash ring data structure (`std::map<float, NodeID>`) with virtual node support
- [ ] Implement shard ID → ring position hash function (SHA256 normalized)
- [ ] Implement heartbeat daemon on each GPU node (configurable `HEARTBEAT_INTERVAL`)
- [ ] Implement timeout-based fault detector with `TIMEOUT_LIMIT` counter per peer node
- [ ] Implement gossip-based failure broadcast and quorum confirmation

### Phase 2 — Checkpoint Integration

- [ ] Define checkpoint shard schema and manifest format
- [ ] Implement checkpoint writer (end-of-epoch flush to central storage)
- [ ] Implement micro-checkpoint writer (configurable step interval)
- [ ] Implement shard cache manager on each GPU node's NVMe
- [ ] Integrate `LD_PRELOAD` interceptor (or equivalent) to redirect I/O through the cache layer

### Phase 3 — Recovery Scheduler

- [ ] Implement `select_resume_mode()` based on epoch progress and node loss ratio
- [ ] Implement shard re-assignment algorithm using updated hash ring
- [ ] Implement barrier synchronization for coordinated epoch rollback
- [ ] Implement work slice broadcast to surviving GPU nodes
- [ ] Test multi-failure scenarios (2, 3, 4 simultaneous node failures)

### Phase 4 — Tuning and Validation

- [ ] Run load distribution simulation (500 trials, vary virtual node count 10–1000)
- [ ] Identify optimal virtual node count for your shard count (expect diminishing returns beyond ~500 virtual nodes per physical node)
- [ ] Measure victim epoch overhead: compare NVMe recaching vs. PFS-redirection baseline
- [ ] Validate that each orphaned shard triggers exactly one central storage access

---

## 7. Configuration Reference

```yaml
# hash_ring_config.yaml

hash_ring:
  virtual_nodes_per_physical: 100      # tune based on shard count; 100 is paper's optimum
  hash_algorithm: sha256
  ring_range: [0.0, 1.0]

fault_detector:
  ttl_seconds: 5                       # must exceed max observed network RTT
  timeout_limit: 3                     # consecutive timeouts before SUSPECTED
  heartbeat_interval_seconds: 1
  suspicion_quorum_pct: 51             # % of peers that must confirm failure

recovery_scheduler:
  hot_resume_threshold: 0.80           # epoch completion % to trigger hot resume
  critical_failure_node_loss_pct: 20   # % node loss to halt and alert
  micro_checkpoint_step_interval: 500  # save micro-checkpoint every N steps
  checkpoint_retention_epochs: 2       # keep last N epoch checkpoints

central_storage:
  base_path: /storage/checkpoints
  shard_size_bytes: 536870912          # 512 MB per shard (tune to model size)
  write_mode: end_of_epoch             # "end_of_epoch" | "micro" | "both"

nvme_cache:
  mount_path: /nvme/cache
  max_cache_size_gb: 3500              # leave headroom on the SSD
  eviction_policy: lru
```

---

## 8. Performance Expectations

Based on the paper's Frontier results (scaled to your environment):

| Scenario | Expected Runtime Overhead vs. No-Failure Baseline |
|---|---|
| No failure (FT overhead only) | ~1–2% (timeout checks, mutex locks, conditional branches) |
| 1 node failure, NVMe recaching | ~12–27% depending on node count |
| 1 node failure, naive PFS redirection | ~32–69% depending on node count |
| Straggler effect at large scale | Contained — minimizing PFS access is the key lever |

The gap between recaching and naive redirection **does not close** as node count scales, due to the straggler effect: even a small number of nodes hitting central storage on every batch forces all other nodes to wait at the synchronization barrier.

---

## 9. Key Design Tradeoffs

| Decision | Chosen Approach | Alternative | Reason |
|---|---|---|---|
| Failure detection | Timeout + quorum | Centralized monitor | Avoids single point of failure in detector |
| Shard redistribution | Hash ring (consistent hashing) | Modulo rehash | Modulo rehash moves all data; hash ring moves only orphaned shards |
| Load balancing | Virtual nodes (100 per physical) | Range partitioning | Range partitioning needs range adjustment on failure; virtual nodes are self-balancing |
| Recovery point | Epoch rollback | Mini-batch rollback | Simpler state; mini-batch rollback requires more frequent micro-checkpoints |
| Central storage access | Once per lost shard | Every epoch per lost shard | Paper's core contribution; 2–3× better runtime under failure |

---

## 10. References

- Lee et al., *Fault-Tolerant Deep Learning Cache with Hash Ring for Load Balancing in HPC Systems*, SC '24 Workshops, 2024. DOI: 10.1109/SCW63240.2024.00176
- Karger et al., *Consistent Hashing and Random Trees*, STOC '97 — foundational consistent hashing paper.
- Khan et al., *HVAC: Removing I/O Bottleneck for Large-Scale Deep Learning Applications*, IEEE CLUSTER 2022 — the base caching system this work extends.
- Source code (FT-Cache): https://github.com/lass-lab/FT-Cache/
