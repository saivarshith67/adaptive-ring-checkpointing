---
phase: 10-fault-detector
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/coci/hashing/fault_detector.py
  - src/coci/hashing/hash_ring.py
  - src/coci/hashing/__init__.py
autonomous: true
requirements:
  - FLTD-01
  - FLTD-02
  - FLTD-03
  - FLTD-04

must_haves:
  truths:
    - "Timeout-based detection marks nodes as SUSPECTED after ttl_seconds * timeout_limit"
    - "Node status correctly transitions ALIVE → SUSPECTED → DEAD"
    - "Gossip broadcast sends heartbeat to all peers via torch.distributed"
    - "Quorum (51%) confirms DEAD status before final state change"
  artifacts:
    - path: "src/coci/hashing/fault_detector.py"
      provides: "FaultDetector with distributed gossip primitives"
      contains: "broadcast_heartbeat, gather_suspicions, async_heartbeat_thread"
    - path: "src/coci/hashing/hash_ring.py"
      provides: "HashRing integration for node registration"
      contains: "register_fault_detector_nodes"
    - path: "src/coci/hashing/__init__.py"
      provides: "Exports for FaultDetector"
      contains: "FaultDetector, NodeStatus"
  key_links:
    - from: "src/coci/hashing/fault_detector.py"
      to: "torch.distributed"
      via: "dist.all_reduce, dist.all_gather"
      pattern: "torch\\.distributed"
    - from: "src/coci/hashing/fault_detector.py"
      to: "hash_ring.HashRing"
      via: "get_nodes() method"
      pattern: "hash_ring.*get_nodes"
---

<objective>
Implement distributed gossip-based failure detection with quorum confirmation.

Purpose: Enable decentralized failure detection using torch.distributed primitives so nodes can detect peer failures without centralized coordinator. This is critical for Phase 11 (Elastic Recaching) to know which shards are orphaned.

Output: Enhanced FaultDetector class with distributed gossip primitives, integrated with HashRing.
</objective>

<context>
@src/coci/hashing/fault_detector.py
@src/coci/hashing/hash_ring.py

**Existing implementation status:**
- FaultDetector already has: ttl_seconds, timeout_limit, heartbeat_interval, NodeStatus enum (ALIVE/SUSPECTED/DEAD), suspicion_quorum_pct=51
- Missing: distributed gossip broadcast (FLTD-03)
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add distributed gossip primitives to FaultDetector</name>
  <files>src/coci/hashing/fault_detector.py</files>
  <action>
    Enhance FaultDetector class with torch.distributed primitives for gossip-based failure broadcast:

    1. Add optional rank/world_size parameters to __init__:
       - rank: int = None (current process rank)
       - world_size: int = None (total processes)

    2. Add broadcast_heartbeat() method:
       - Use dist.all_reduce to aggregate heartbeat signals from all peers
       - Returns dict of {node_id: aggregated_heartbeat_count}
       - Handle non-distributed mode gracefully (return local only)

    3. Add gather_suspicions() method:
       - Use dist.all_gather to collect suspicion votes from all ranks
       - Aggregate votes across all nodes
       - Return dict of {node_id: total_suspicion_votes}

    4. Add is_distributed() property to check if running in distributed mode

    **IMPORTANT:** The existing FaultDetector already implements:
    - FLTD-01: check_timeout() method with ttl_seconds and timeout_limit
    - FLTD-02: NodeStatus enum with ALIVE, SUSPECTED, DEAD
    - FLTD-04: register_suspicion() with quorum calculation (51%)
    - Only FLTD-03 requires enhancement
  </action>
  <verify>
    <automated>
python -c "
import torch
import torch.distributed as dist
import sys

# Test non-distributed mode
from src.coci.hashing.fault_detector import FaultDetector, NodeStatus

fd = FaultDetector(ttl_seconds=5.0, timeout_limit=3)
fd.register_node('rank:0')
fd.register_node('rank:1')

# Verify FLTD-01: TTL-based timeout detection
fd.mark_alive('rank:0')
import time
time.sleep(0.1)
fd.check_timeout('rank:0')
print(f'Status after check: {fd.get_status(\"rank:0\")}')

# Verify FLTD-02: Node status tracking
assert fd.get_status('rank:0') == NodeStatus.ALIVE

# Verify FLTD-04: Quorum-based confirmation
fd.register_suspicion('rank:0', 'rank:1')
fd.register_suspicion('rank:0', 'rank:2')
fd.register_suspicion('rank:0', 'rank:3')
# With 51% quorum, 3/4 votes = DEAD
assert fd.get_status('rank:0') == NodeStatus.DEAD

# Verify is_distributed property
assert fd.is_distributed == False

print('FLTD-01, FLTD-02, FLTD-04 verified')
"
    </automated>
  </verify>
  <done>
    FaultDetector has broadcast_heartbeat() and gather_suspicions() methods that use torch.distributed.
    Non-distributed mode works correctly (returns local data only).
    FLTD-01, FLTD-02, FLTD-04 continue to work as before.
  </done>
</task>

<task type="auto">
  <name>Task 2: Add async heartbeat thread for distributed gossip</name>
  <files>src/coci/hashing/fault_detector.py</files>
  <action>
    Add async heartbeat mechanism to FaultDetector:

    1. Add start_heartbeat_thread(rank, world_size) method:
       - Spawns daemon thread running _heartbeat_loop()
       - Calls broadcast_heartbeat() every heartbeat_interval seconds
       - Updates local node status based on aggregated peer heartbeats
       - Stops gracefully on stop_heartbeat_thread()

    2. Add stop_heartbeat_thread() method:
       - Sets stop_event
       - Waits for thread to join (max 2 seconds)

    3. Add _heartbeat_loop() private method:
       - Runs in background thread
       - Periodically broadcasts local heartbeat
       - Aggregates peer heartbeats
       - Detects timeouts and registers suspicions

    4. Add get_status_summary() method:
       - Returns dict with alive/suspected/dead counts
       - Useful for monitoring/debugging
  </action>
  <verify>
    <automated>
python -c "
from src.coci.hashing.fault_detector import FaultDetector
import time

fd = FaultDetector(ttl_seconds=1.0, timeout_limit=2, heartbeat_interval=0.5)
fd.register_node('rank:0')

# Start heartbeat thread (non-distributed mode)
fd.start_heartbeat_thread(rank=0, world_size=1)
time.sleep(1.5)

# Get status summary
summary = fd.get_status_summary()
print(f'Status summary: {summary}')

# Stop thread
fd.stop_heartbeat_thread()

print('Async heartbeat thread working')
"
    </automated>
  </verify>
  <done>
    Background thread periodically broadcasts heartbeats without blocking training loop.
    get_status_summary() returns alive/suspected/dead counts.
  </done>
</task>

<task type="auto">
  <name>Task 3: Integrate FaultDetector with HashRing + exports</name>
  <files>src/coci/hashing/fault_detector.py, src/coci/hashing/hash_ring.py, src/coci/hashing/__init__.py</files>
  <action>
    Integration and exports:

    1. Update src/coci/hashing/__init__.py to export:
       - from .fault_detector import FaultDetector, NodeStatus, ElasticRecaching

    2. Add register_nodes_from_hash_ring(fault_detector, hash_ring) helper:
       - In hash_ring.py or fault_detector.py
       - Iterates hash_ring.get_nodes() and calls fault_detector.register_node() for each

    3. Add example integration in fault_detector.py docstring showing:
       - HashRing creation
       - ShardManager creation
       - FaultDetector initialization
       - Node registration from hash ring
  </action>
  <verify>
    <automated>
python -c "
from src.coci.hashing import FaultDetector, NodeStatus
from src.coci.hashing import HashRing, ShardManager

# Integration test
hash_ring = HashRing(virtual_nodes_per_physical=10)
hash_ring.add_node('gpu:0')
hash_ring.add_node('gpu:1')

fault_detector = FaultDetector()

# Register nodes from hash ring
for node_id in hash_ring.get_nodes():
    fault_detector.register_node(node_id)

alive = fault_detector.get_alive_nodes()
print(f'Registered nodes: {alive}')
assert len(alive) == 2

print('HashRing integration working')
"
    </automated>
  </verify>
  <done>
    FaultDetector properly exports from coci.hashing package.
    HashRing nodes can be registered with FaultDetector.
  </done>
</task>

</tasks>

<verification>
After implementation:
1. Import and test FaultDetector in non-distributed mode - PASS
2. Start/stop heartbeat thread without blocking - PASS
3. Integrate with HashRing for node registration - PASS
4. All 4 requirements (FLTD-01 through FLTD-04) functional
</verification>

<success_criteria>
Phase 10 complete when:
- FaultDetector has distributed gossip methods (broadcast_heartbeat, gather_suspicions)
- Async heartbeat thread works without blocking training
- All 4 requirements satisfied: FLTD-01, FLTD-02, FLTD-03, FLTD-04
- Integration with HashRing works for node registration
</success_criteria>

<output>
After completion, create .planning/phases/10-fault-detector/10-fault-detector-01-SUMMARY.md
</output>