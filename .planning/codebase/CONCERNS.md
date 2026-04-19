# Codebase Concerns

**Analysis Date:** 2026-04-18

## Tech Debt

### Incomplete Main Entry Point
- **Issue:** `src/main.py` is a stub with only "Hello world" - no actual training or functionality
- **Files:** `src/main.py`
- **Impact:** Cannot run training directly from the main entry point; must use scripts in `scripts/` directory
- **Fix approach:** Implement proper CLI and training loop in main.py, or remove if truly unused

### Hardcoded Configuration Values
- **Issue:** Configuration defaults are hardcoded in `src/coci/config.py` rather than being configurable
- **Files:** `src/coci/config.py` (lines 5-10)
- **Impact:** Cannot change dataset, model, or training parameters without modifying source code
- **Fix approach:** Load all defaults from config YAML files; use environment variables for overrides

### Checkpoint Manager Metric Tracking
- **Issue:** CheckpointManager tracks `num_checkpoints` and `total_checkpoint_time` but only prints them; not exposed for external monitoring
- **Files:** `src/coci/checkpointing/checkpoint_manager.py` (lines 13-15)
- **Impact:** Cannot programmatically access checkpoint metrics for analytics
- **Fix approach:** Add getter methods: `get_num_checkpoints()` and `get_total_checkpoint_time()`

### Incomplete Hash Ring Shard Management
- **Issue:** `ShardManager.register_shard()` uses fallback `or self.hash_ring.get_nodes()[0]` which assumes at least one node exists
- **Files:** `src/coci/hashing/hash_ring.py` (line 82)
- **Impact:** Will fail with IndexError if hash ring is empty
- **Fix approach:** Raise explicit error if no nodes available, or add default node creation

## Known Bugs

### Recovery Scheduler Return Values
- **Issue:** `compute_rollback()` returns `self.resume_from_epoch` which is always `last_clean_epoch` regardless of resume mode - HOT_RESUME, COLD_RESUME, and CRITICAL_FAILURE all return the same value
- **Files:** `src/coci/hashing/recovery_scheduler.py` (lines 45-55)
- **Impact:** Resume mode selection has no practical effect on which checkpoint is loaded
- **Fix approach:** Differentiate resume epochs based on mode: HOT_RESUME could use last checkpoint, COLD_RESUME should use epoch-1, CRITICAL_FAILURE should restart from epoch 0

### ElasticRecaching Status Updates
- **Issue:** `recache_shard()` updates status twice: first to "CACHING" then immediately to "RE-CACHED", skipping actual caching logic
- **Files:** `src/coci/hashing/fault_detector.py` (lines 121-122)
- **Impact:** Shards marked for recaching but actual caching never executes; orphan handling is incomplete
- **Fix approach:** Implement actual caching/replication logic between these status updates

### FaceForensics Dataset Video Index Building
- **Issue:** Video index building assumes specific directory structure; may fail silently if structure differs
- **Files:** `src/coci/data_ingestor/faceforensics.py` (_build_from_manifest, _build_from_directories)
- **Impact:** Dataset returns 0 videos without warning if directory structure doesn't match expected pattern
- **Fix approach:** Add validation/assertion with clear error message; log what directories were searched

### Fault Detector Status Transitions
- **Issue:** Node can go from ALIVE directly to SUSPECTED via `check_timeout()`, but also to DEAD via `register_suspicion()`; both transitions can happen independently
- **Files:** `src/coci/hashing/fault_detector.py` (lines 66-76, 48-64)
- **Impact:** Node could be both SUSPECTED and DEAD simultaneously depending on call order; race condition
- **Fix approach:** Use state machine with explicit transition rules; add locking around status changes

## Security Considerations

### Kaggle API Credentials
- **Issue:** Uses `kagglehub` for dataset download which requires Kaggle credentials; no handling for credential issues
- **Files:** `src/coci/data_ingestor/faceforensics.py` (line 53)
- **Impact:** Training fails silently if Kaggle credentials not configured; no clear error message
- **Recommendations:** Add try/except with clear error message about Kaggle credentials requirement

### No Input Validation on Config
- **Issue:** `load_config()` uses dataclass without validation; negative epochs or batch_size=0 would cause cryptic errors later
- **Files:** `src/coci/config.py` (lines 28-31)
- **Impact:** Configuration errors caught late in training loop rather than at startup
- **Recommendations:** Add __post_init__ validation to Config dataclass

### Checkpoint Loading Security
- **Issue:** `torch.load()` used without specifying weights_only=True; PyTorch checkpoints could execute arbitrary code
- **Files:** `src/coci/checkpointing/checkpoint_manager.py` (line 96)
- **Impact:** Loading untrusted checkpoints is a security risk
- **Recommendations:** Use `torch.load(path, map_location=device, weights_only=True)` unless older checkpoints need to be compatible

## Performance Bottlenecks

### Data Loading Performance
- **Issue:** Default num_workers=0 in dev config; no parallel data loading
- **Files:** `configs/dev.yaml` (line 6)
- **Impact:** CPU-bound data preprocessing becomes training bottleneck
- **Improvement path:** Increase num_workers to 4+ for production; add automatic worker count detection

### Checkpoint Filename Parsing
- **Issue:** `load_latest()` uses string splitting: `max(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))` - fragile filename parsing
- **Files:** `src/coci/checkpointing/checkpoint_manager.py` (line 92)
- **Impact:** Checkpoint file with unexpected format causes ValueError
- **Improvement path:** Use regex or structured naming with metadata JSON sidecar

### MTCNN Face Detection
- **Issue:** On-the-fly face detection is extremely slow; no warning about performance impact
- **Files:** `src/coci/data_ingestor/faceforensics.py` (_detect_face_live)
- **Impact:** Training can be 10-100x slower with live detection
- **Improvement path:** Add performance warning in logs when using live detection

### Hash Ring Re-sorting
- **Issue:** Every `add_node()` and `remove_node()` triggers full re-sort: `self.sorted_positions = sorted(self.ring.keys())`
- **Files:** `src/coci/hashing/hash_ring.py` (lines 30, 40)
- **Impact:** O(n log n) per add/remove; problematic with many nodes
- **Improvement path:** Use sortedcontainers or maintain sorted position during insertions

## Fragile Areas

### Distributed Setup Error Handling
- **Issue:** `setup_distributed()` silently treats missing torchrun env vars as single-process mode
- **Files:** `src/coci/distributed.py` (lines 121-125)
- **Why fragile:** Code runs in single-process mode without warning; DDP training silently becomes single-GPU
- **Safe modification:** Add explicit `is_distributed=False` flag parameter; warn if explicitly passed but env vars missing

### Checkpoint Manager DDP Detection
- **Issue:** Uses `dist.get_rank()` to determine if rank 0 but called at module level; may be called before dist init
- **Files:** `src/coci/checkpointing/checkpoint_manager.py` (line 29)
- **Why fragile:** Could be called before distributed init, returning incorrect rank, causing all ranks to save or no ranks to save
- **Safe modification:** Pass rank explicitly or use `setup_distributed()` return value

### Training Scripts Exception Handling
- **Issue:** `train_faceforensics.py` catches Exception at epoch level but uses `val_loss if "val_loss" in dir()` - fragile conditional
- **Files:** `scripts/train_faceforensics.py` (line 1026)
- **Why fragile:** Will use 0.0 if exception happens before val_loss is defined; may not save useful checkpoint
- **Safe modification:** Initialize val_loss before try block; use more specific exception types

### Config Dataclass Immutability
- **Issue:** Config uses dataclass (mutable) but treated as immutable (no setter usage observed)
- **Files:** `src/coci/config.py` (lines 13-26)
- **Why fragile:** Could be modified mid-training causing inconsistent state
- **Safe modification:** Use frozen=True or convert to immutable after creation

## Scaling Limits

### Distributed Process Group Backend
- **Issue:** Hardcoded "nccl" backend; no fallback for CPU-only or multi-machine
- **Files:** `src/coci/distributed.py` (line 95)
- **Current capacity:** Single-node multi-GPU only
- **Limit:** Multi-node training requires GLOO or custom init
- **Scaling path:** Add backend detection: GLOO for CPU/multi-node, NCCL for single-node GPU

### Checkpoint Storage
- **Issue:** All checkpoints saved to single directory with sequential naming; no cleanup or retention policy
- **Files:** `src/coci/checkpointing/checkpoint_manager.py` (line 36)
- **Current capacity:** Limited by disk space only
- **Limit:** Can fill disk after many experiments
- **Scaling path:** Add checkpoint cleanup, best-only saving, or incremental checkpoint merging

### Hash Ring Node Count
- **Issue:** `ShardedManager` uses in-memory dict; no persistence or synchronization
- **Files:** `src/coci/hashing/hash_ring.py` (line 73)
- **Current capacity:** Single process only
- **Limit:** Node state lost on restart; no distributed coordination
- **Scaling path:** Integrate with etcd/Consul for distributed node state

## Dependencies at Risk

### KaggleHub Dependency
- **Issue:** Dataset download relies on kagglehub package; no alternative download method
- **Impact:** If kagglehub is deprecated or rate-limited, dataset download fails
- **Migration plan:** Implement direct URL download as fallback; support manual dataset placement

### Facenet-Pytorch
- **Issue:** MTCNN from facenet-pytorch used for face detection; optional import with fallback
- **Impact:** With fallback, accuracy may degrade (falls back to resize)
- **Migration plan:** Add explicit dependency check at startup; recommend alternatives (RetinaFace, MediaPipe)

### PyTorch Version
- **Issue:** Requires torch>=2.10.0 which is a very new version
- **Impact:** May not be available in all environments; compatibility issues
- **Migration plan:** Test with current LTS (2.5.x); update minimum versions as needed

## Missing Critical Features

### Test Coverage
- **Issue:** No test files found in codebase
- **Problem:** No verification that components work correctly
- **Blocks:** Safe refactoring; bug detection before production

### Logging System
- **Issue:** Uses print() statements scattered throughout; no structured logging
- **Problem:** Cannot filter, redirect, or analyze logs programmatically
- **Blocks:** Production debugging; observability

### Configuration Validation
- **Issue:** No validation that config values are reasonable before training starts
- **Problem:** Invalid configs cause late failures
- **Blocks:** Reliable automation; CI/CD

### Checkpoint Verification
- **Issue:** Loaded checkpoints not verified; corrupted checkpoint causes cryptic errors
- **Problem:** Training resumes from corrupted state silently
- **Blocks:** Reliable recovery; checkpoint integrity

### Graceful Shutdown
- **Issue:** No signal handling for SIGINT/SIGTERM
- **Problem:** Checkpointing may not complete on interrupt
- **Blocks:** Reliable production runs

## Test Coverage Gaps

### Unit Tests
- **What's not tested:** HashRing operations, Checkpoint strategies, Fault detection logic
- **Files:** `src/coci/hashing/`, `src/coci/checkpointing/`
- **Risk:** Logic errors in distributed coordination go undetected
- **Priority:** High

### Integration Tests
- **What's not tested:** Full training loop end-to-end
- **Files:** Training scripts
- **Risk:** Integration issues between components
- **Priority:** Medium

### Data Pipeline Tests
- **What's not tested:** Dataset loading, face detection
- **Files:** `src/coci/data_ingestor/`
- **Risk:** Data issues cause silent accuracy degradation
- **Priority:** Medium

---

*Concerns audit: 2026-04-18*