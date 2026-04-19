"""Comprehensive metrics collection for training experiments."""

import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from collections import defaultdict


@dataclass
class EpochMetrics:
    """Metrics for a single epoch."""
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    epoch_duration_sec: float
    checkpoint_count: int = 0
    checkpoint_time_sec: float = 0.0
    checkpoint_size_mb: float = 0.0
    

@dataclass
class ConvergenceMetrics:
    """Convergence-aware scheduler metrics."""
    current_interval_sec: float = 0.0
    theta1: Optional[float] = None
    theta2: Optional[float] = None
    loss_fit_error: Optional[float] = None
    segments_since_restart: int = 0
    total_convergence_checkpoints: int = 0


@dataclass
class HashRingMetrics:
    """Hash-ring checkpoint manager metrics."""
    total_shards: int = 0
    local_cache_hits: int = 0
    local_cache_misses: int = 0
    local_cache_loads: int = 0
    central_storage_loads: int = 0
    cache_write_count: int = 0
    cache_size_mb: float = 0.0
    max_cache_size_mb: float = 0.0
    shards_owned: int = 0
    orphaned_shards: int = 0
    reassigned_shards: int = 0
    recached_shards: int = 0
    shard_recovery_count: int = 0
    recovery_time_sec: float = 0.0
    last_load_source: Optional[str] = None


@dataclass
class FaultMetrics:
    """Fault injection and recovery metrics."""
    injected_faults: int = 0
    detected_faults: int = 0
    recovery_attempts: int = 0
    recovered_successfully: int = 0
    recovery_failures: int = 0
    total_recovery_time_sec: float = 0.0
    checkpoint_integrity_failures: int = 0
    runtime_fault_triggered: bool = False
    runtime_fault_checkpoint_id: Optional[str] = None
    runtime_fault_epoch: Optional[int] = None
    runtime_fault_global_step: Optional[int] = None
    runtime_fault_wall_time_sec: Optional[float] = None
    resume_attempted: bool = False
    resume_succeeded: bool = False
    resumed_from_epoch: Optional[int] = None
    resumed_checkpoint_id: Optional[str] = None
    resumed_checkpoint_path: Optional[str] = None
    resume_load_source: Optional[str] = None
    time_to_resume_sec: float = 0.0
    rollback_epochs: float = 0.0
    rollback_steps: int = 0
    lost_work_sec: float = 0.0
    failure_reason: Optional[str] = None
    last_completed_epoch: int = 0
    last_completed_global_step: int = 0


@dataclass
class DistributedMetrics:
    """Distributed training metrics."""
    world_size: int = 1
    num_syncs: int = 0
    total_sync_time_sec: float = 0.0
    comm_overhead_percent: float = 0.0


@dataclass
class ExperimentSummary:
    """Overall experiment summary."""
    experiment_name: str
    training_mode: str
    dataset: str
    model: str
    epochs: int
    batch_size: int
    
    # Timing metrics
    total_training_time_sec: float = 0.0
    total_checkpoint_time_sec: float = 0.0
    total_checkpoint_count: int = 0
    checkpoint_overhead_percent: float = 0.0
    
    # Training performance
    best_train_loss: float = float('inf')
    best_val_loss: float = float('inf')
    best_val_accuracy: float = 0.0
    final_train_loss: float = 0.0
    final_val_loss: float = 0.0
    final_val_accuracy: float = 0.0
    
    # Data throughput
    total_samples_trained: int = 0
    samples_per_sec: float = 0.0
    
    # Sub-metrics
    convergence_metrics: ConvergenceMetrics = field(default_factory=ConvergenceMetrics)
    hash_ring_metrics: HashRingMetrics = field(default_factory=HashRingMetrics)
    fault_metrics: FaultMetrics = field(default_factory=FaultMetrics)
    distributed_metrics: DistributedMetrics = field(default_factory=DistributedMetrics)


class MetricsCollector:
    """Collects metrics throughout training for later export and analysis."""
    
    def __init__(self, experiment_name: str, training_mode: str, 
                 dataset: str, model: str, epochs: int, batch_size: int):
        """Initialize metrics collector.
        
        Args:
            experiment_name: Name of the experiment (e.g., 'faceforensics_exp1')
            training_mode: Mode of training (epoch, convergence, hash-ring-epoch, convergence-hash-ring)
            dataset: Name of dataset (e.g., 'faceforensics', 'cifar100')
            model: Model name (e.g., 'efficientnet-b0')
            epochs: Total number of epochs
            batch_size: Batch size
        """
        self.experiment_name = experiment_name
        self.training_mode = training_mode
        self.dataset = dataset
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Timing
        self.start_time = time.time()
        self._training_start_time = time.time()
        self._checkpoint_times: List[float] = []
        
        # Per-epoch tracking
        self.epoch_metrics: List[EpochMetrics] = []
        
        # Training stats
        self._train_losses: List[float] = []
        self._train_accuracies: List[float] = []
        self._val_losses: List[float] = []
        self._val_accuracies: List[float] = []
        
        # Checkpoint tracking
        self._checkpoint_sizes: List[float] = []
        self._checkpoint_count = 0
        self._total_checkpoint_time = 0.0
        
        # Sub-systems
        self.convergence_history: List[ConvergenceMetrics] = []
        self.hash_ring_history: List[HashRingMetrics] = []
        self.fault_history: List[FaultMetrics] = []
        
        # Current state for active tracking
        self.current_convergence = ConvergenceMetrics()
        self.current_hash_ring = HashRingMetrics()
        self.current_faults = FaultMetrics()
        self.current_distributed = DistributedMetrics()
        
        # Batch tracking
        self._batch_count = 0
        self._epoch_start_time = None
    
    # ============================================
    # Epoch Management
    # ============================================
    
    def start_epoch(self, epoch: int) -> None:
        """Mark the start of an epoch."""
        self._epoch_start_time = time.time()
        
    def end_epoch(self, epoch: int, train_loss: float, train_acc: float,
                  val_loss: float, val_acc: float) -> None:
        """Record metrics for completed epoch.
        
        Args:
            epoch: Epoch number (0-indexed)
            train_loss: Average training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
        """
        epoch_duration = time.time() - self._epoch_start_time if self._epoch_start_time else 0.0
        
        epoch_metric = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_acc,
            val_loss=val_loss,
            val_accuracy=val_acc,
            epoch_duration_sec=epoch_duration,
        )
        
        self.epoch_metrics.append(epoch_metric)
        self._train_losses.append(train_loss)
        self._train_accuracies.append(train_acc)
        self._val_losses.append(val_loss)
        self._val_accuracies.append(val_acc)
    
    # ============================================
    # Checkpoint Metrics
    # ============================================
    
    def record_checkpoint(self, checkpoint_size_mb: float, save_time_sec: float,
                          checkpoint_id: Optional[str] = None) -> None:
        """Record checkpoint metrics.
        
        Args:
            checkpoint_size_mb: Size of checkpoint in MB
            save_time_sec: Time taken to save checkpoint
            checkpoint_id: Optional checkpoint identifier
        """
        self._checkpoint_count += 1
        self._checkpoint_sizes.append(checkpoint_size_mb)
        self._total_checkpoint_time += save_time_sec
        self._checkpoint_times.append(save_time_sec)
        
        if self.epoch_metrics:
            self.epoch_metrics[-1].checkpoint_count += 1
            self.epoch_metrics[-1].checkpoint_time_sec += save_time_sec
            self.epoch_metrics[-1].checkpoint_size_mb = checkpoint_size_mb
    
    def record_batch_checkpoint(self, batch_id: int, checkpoint_size_mb: float,
                                save_time_sec: float) -> None:
        """Record in-epoch checkpoint metrics (e.g., convergence-based).
        
        Args:
            batch_id: Batch identifier
            checkpoint_size_mb: Size of checkpoint in MB
            save_time_sec: Time taken to save
        """
        self._checkpoint_count += 1
        self._checkpoint_sizes.append(checkpoint_size_mb)
        self._total_checkpoint_time += save_time_sec
        self._checkpoint_times.append(save_time_sec)
    
    # ============================================
    # Convergence Scheduler Metrics
    # ============================================
    
    def update_convergence_metrics(self, current_interval_sec: float,
                                   theta1: Optional[float] = None,
                                   theta2: Optional[float] = None,
                                   loss_fit_error: Optional[float] = None) -> None:
        """Update convergence-aware scheduler metrics.
        
        Args:
            current_interval_sec: Current checkpoint interval in seconds
            theta1: Exponential decay parameter
            theta2: Exponential offset parameter
            loss_fit_error: Error in loss function fit
        """
        self.current_convergence.current_interval_sec = current_interval_sec
        if theta1 is not None:
            self.current_convergence.theta1 = theta1
        if theta2 is not None:
            self.current_convergence.theta2 = theta2
        if loss_fit_error is not None:
            self.current_convergence.loss_fit_error = loss_fit_error
    
    def increment_convergence_checkpoint(self) -> None:
        """Increment convergence-based checkpoint counter."""
        self.current_convergence.total_convergence_checkpoints += 1
    
    def record_convergence_segment_restart(self) -> None:
        """Record that convergence fitting segment restarted (POF)."""
        self.current_convergence.segments_since_restart += 1
        self.convergence_history.append(ConvergenceMetrics(**asdict(self.current_convergence)))
    
    # ============================================
    # Hash Ring Metrics
    # ============================================
    
    def update_hash_ring_metrics(self, total_shards: int, cache_size_mb: float,
                                 max_cache_mb: float, shards_owned: int) -> None:
        """Update hash-ring checkpoint metrics.
        
        Args:
            total_shards: Total number of shards in ring
            cache_size_mb: Current cache size in MB
            max_cache_mb: Maximum cache size in MB
            shards_owned: Number of shards owned by this node
        """
        self.current_hash_ring.total_shards = total_shards
        self.current_hash_ring.cache_size_mb = cache_size_mb
        self.current_hash_ring.max_cache_size_mb = max_cache_mb
        self.current_hash_ring.shards_owned = shards_owned
    
    def increment_cache_hit(self) -> None:
        """Record local cache hit."""
        self.current_hash_ring.local_cache_hits += 1
        self.current_hash_ring.local_cache_loads += 1
        self.current_hash_ring.last_load_source = "local-cache"

    def increment_cache_miss(self) -> None:
        """Record local cache miss."""
        self.current_hash_ring.local_cache_misses += 1
        self.current_hash_ring.last_load_source = "central-storage"

    def record_hash_ring_load(self, source: str) -> None:
        """Record where a checkpoint was loaded from."""
        self.current_hash_ring.last_load_source = source
        if source == "local-cache":
            self.current_hash_ring.local_cache_hits += 1
            self.current_hash_ring.local_cache_loads += 1
        elif source == "central-storage":
            self.current_hash_ring.local_cache_misses += 1
            self.current_hash_ring.central_storage_loads += 1

    def record_hash_ring_cache_write(self) -> None:
        """Record a shard being written to local cache."""
        self.current_hash_ring.cache_write_count += 1

    def record_hash_ring_reassignment(self, orphaned_shards: int, reassigned_shards: int) -> None:
        """Record shard reassignment after a failure."""
        self.current_hash_ring.orphaned_shards += orphaned_shards
        self.current_hash_ring.reassigned_shards += reassigned_shards

    def record_recached_shards(self, count: int) -> None:
        """Record how many orphaned shards were re-cached."""
        self.current_hash_ring.recached_shards += count
    
    def record_shard_recovery(self, recovery_time_sec: float) -> None:
        """Record shard recovery event.
        
        Args:
            recovery_time_sec: Time taken to recover shard
        """
        self.current_hash_ring.shard_recovery_count += 1
        self.current_hash_ring.recovery_time_sec += recovery_time_sec
    
    # ============================================
    # Fault Metrics
    # ============================================
    
    def increment_injected_fault(self) -> None:
        """Record injected fault."""
        self.current_faults.injected_faults += 1
    
    def increment_detected_fault(self) -> None:
        """Record detected fault."""
        self.current_faults.detected_faults += 1

    def record_runtime_fault(
        self,
        checkpoint_id: str,
        epoch: int,
        global_step: int,
        wall_time_sec: Optional[float] = None,
    ) -> None:
        """Record a runtime fault injection event."""
        self.current_faults.injected_faults += 1
        self.current_faults.runtime_fault_triggered = True
        self.current_faults.runtime_fault_checkpoint_id = checkpoint_id
        self.current_faults.runtime_fault_epoch = epoch
        self.current_faults.runtime_fault_global_step = global_step
        self.current_faults.runtime_fault_wall_time_sec = wall_time_sec

    def record_resume_attempt(self) -> None:
        """Record that a resume was attempted."""
        self.current_faults.resume_attempted = True
        self.current_faults.recovery_attempts += 1

    def record_resume_success(
        self,
        resumed_from_epoch: int,
        checkpoint_id: Optional[str],
        checkpoint_path: Optional[str],
        load_source: Optional[str],
        time_to_resume_sec: float = 0.0,
        rollback_epochs: float = 0.0,
        rollback_steps: int = 0,
        lost_work_sec: float = 0.0,
    ) -> None:
        """Record a successful resume event and its rollback characteristics."""
        self.current_faults.resume_succeeded = True
        self.current_faults.resumed_from_epoch = resumed_from_epoch
        self.current_faults.resumed_checkpoint_id = checkpoint_id
        self.current_faults.resumed_checkpoint_path = checkpoint_path
        self.current_faults.resume_load_source = load_source
        self.current_faults.time_to_resume_sec += time_to_resume_sec
        self.current_faults.rollback_epochs = rollback_epochs
        self.current_faults.rollback_steps = rollback_steps
        self.current_faults.lost_work_sec = lost_work_sec

    def record_failure_event(
        self,
        reason: str,
        epoch: int,
        global_step: int,
    ) -> None:
        """Record the last observed failure state."""
        self.current_faults.failure_reason = reason
        self.current_faults.last_completed_epoch = epoch
        self.current_faults.last_completed_global_step = global_step

    def record_recovery(self, success: bool, recovery_time_sec: float) -> None:
        """Record recovery attempt.
        
        Args:
            success: Whether recovery succeeded
            recovery_time_sec: Time taken to recover
        """
        if success:
            self.current_faults.recovered_successfully += 1
        else:
            self.current_faults.recovery_failures += 1
        self.current_faults.total_recovery_time_sec += recovery_time_sec
    
    def increment_checkpoint_integrity_failure(self) -> None:
        """Record checkpoint integrity check failure."""
        self.current_faults.checkpoint_integrity_failures += 1
    
    # ============================================
    # Distributed Training Metrics
    # ============================================
    
    def set_distributed_config(self, world_size: int) -> None:
        """Set distributed training configuration.
        
        Args:
            world_size: Number of GPUs/ranks
        """
        self.current_distributed.world_size = world_size
    
    def record_synchronization(self, sync_time_sec: float) -> None:
        """Record collective synchronization operation.
        
        Args:
            sync_time_sec: Time spent in synchronization
        """
        self.current_distributed.num_syncs += 1
        self.current_distributed.total_sync_time_sec += sync_time_sec
    
    # ============================================
    # Batch Processing
    # ============================================
    
    def increment_batch_count(self, samples: int = 1) -> None:
        """Increment batch counter.
        
        Args:
            samples: Number of samples processed in batch
        """
        self._batch_count += 1

    def mark_progress(self, epoch: int, global_step: int) -> None:
        """Record the latest known training progress for recovery metrics."""
        self.current_faults.last_completed_epoch = epoch
        self.current_faults.last_completed_global_step = global_step
    
    # ============================================
    # Finalization & Summary
    # ============================================
    
    def finalize(self) -> ExperimentSummary:
        """Generate final summary metrics."""
        total_training_time = time.time() - self.start_time
        
        # Calculate best and final metrics
        best_train_loss = min(self._train_losses) if self._train_losses else float('inf')
        best_val_loss = min(self._val_losses) if self._val_losses else float('inf')
        best_val_accuracy = max(self._val_accuracies) if self._val_accuracies else 0.0
        final_train_loss = self._train_losses[-1] if self._train_losses else 0.0
        final_val_loss = self._val_losses[-1] if self._val_losses else 0.0
        final_val_accuracy = self._val_accuracies[-1] if self._val_accuracies else 0.0
        
        # Calculate checkpoint overhead
        checkpoint_overhead_percent = (self._total_checkpoint_time / total_training_time * 100.0) if total_training_time > 0 else 0.0
        
        # Calculate communication overhead
        comm_overhead_percent = (self.current_distributed.total_sync_time_sec / total_training_time * 100.0) if total_training_time > 0 else 0.0
        self.current_distributed.comm_overhead_percent = comm_overhead_percent
        
        summary = ExperimentSummary(
            experiment_name=self.experiment_name,
            training_mode=self.training_mode,
            dataset=self.dataset,
            model=self.model,
            epochs=self.epochs,
            batch_size=self.batch_size,
            total_training_time_sec=total_training_time,
            total_checkpoint_time_sec=self._total_checkpoint_time,
            total_checkpoint_count=self._checkpoint_count,
            checkpoint_overhead_percent=checkpoint_overhead_percent,
            best_train_loss=best_train_loss,
            best_val_loss=best_val_loss,
            best_val_accuracy=best_val_accuracy,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            final_val_accuracy=final_val_accuracy,
            convergence_metrics=self.current_convergence,
            hash_ring_metrics=self.current_hash_ring,
            fault_metrics=self.current_faults,
            distributed_metrics=self.current_distributed,
        )
        
        return summary
    
    def get_epoch_metrics_list(self) -> List[Dict[str, Any]]:
        """Get all epoch metrics as dictionaries for export."""
        return [asdict(em) for em in self.epoch_metrics]
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current state snapshot for debugging."""
        return {
            'experiment_name': self.experiment_name,
            'training_mode': self.training_mode,
            'epochs_completed': len(self.epoch_metrics),
            'total_training_time_sec': time.time() - self.start_time,
            'total_checkpoint_time_sec': self._total_checkpoint_time,
            'checkpoint_count': self._checkpoint_count,
            'convergence_metrics': asdict(self.current_convergence),
            'hash_ring_metrics': asdict(self.current_hash_ring),
            'fault_metrics': asdict(self.current_faults),
            'distributed_metrics': asdict(self.current_distributed),
        }
