"""Export metrics to various formats for analysis and plotting."""

import json
import csv
import os
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List, Any, Optional
from datetime import datetime

from .collector import MetricsCollector, ExperimentSummary


class MetricsExporter:
    """Export experiment metrics in multiple formats."""
    
    def __init__(self, base_export_dir: str = "./experiment_results"):
        """Initialize exporter.
        
        Args:
            base_export_dir: Base directory for all exports
        """
        self.base_export_dir = Path(base_export_dir)
        self.base_export_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_timestamp(self) -> str:
        """Get ISO format timestamp."""
        return datetime.now().isoformat()
    
    # ============================================
    # JSON Export (Per-Run Summary)
    # ============================================
    
    def export_summary_json(self, collector: MetricsCollector, 
                            output_path: Optional[str] = None) -> Path:
        """Export experiment summary as JSON.
        
        Args:
            collector: MetricsCollector instance with finalized metrics
            output_path: Custom output path (optional)
            
        Returns:
            Path to exported file
        """
        summary = collector.finalize()
        
        if output_path is None:
            exp_dir = self.base_export_dir / summary.experiment_name
            exp_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = exp_dir / f"{summary.training_mode}_{timestamp}.json"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'timestamp': self._get_timestamp(),
            'summary': asdict(summary),
            'epoch_metrics': collector.get_epoch_metrics_list(),
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return output_path
    
    # ============================================
    # JSONL Export (Aggregated Results)
    # ============================================
    
    def export_summary_jsonl(self, collector: MetricsCollector, 
                            output_path: Optional[str] = None) -> Path:
        """Export experiment summary as JSONL (one line per experiment).
        
        Good for aggregating multiple runs for statistical analysis.
        
        Args:
            collector: MetricsCollector instance with finalized metrics
            output_path: Custom output path (optional)
            
        Returns:
            Path to exported file
        """
        summary = collector.finalize()
        
        if output_path is None:
            output_path = self.base_export_dir / "experiment_summaries.jsonl"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Flatten summary to single-level dict for JSONL format
        data = {
            'timestamp': self._get_timestamp(),
            'experiment_name': summary.experiment_name,
            'training_mode': summary.training_mode,
            'dataset': summary.dataset,
            'model': summary.model,
            'epochs': summary.epochs,
            'batch_size': summary.batch_size,
            'total_training_time_sec': summary.total_training_time_sec,
            'total_checkpoint_time_sec': summary.total_checkpoint_time_sec,
            'total_checkpoint_count': summary.total_checkpoint_count,
            'checkpoint_overhead_percent': summary.checkpoint_overhead_percent,
            'best_train_loss': summary.best_train_loss,
            'best_val_loss': summary.best_val_loss,
            'best_val_accuracy': summary.best_val_accuracy,
            'final_train_loss': summary.final_train_loss,
            'final_val_loss': summary.final_val_loss,
            'final_val_accuracy': summary.final_val_accuracy,
            # Convergence metrics
            'convergence_interval_sec': summary.convergence_metrics.current_interval_sec,
            'convergence_checkpoints': summary.convergence_metrics.total_convergence_checkpoints,
            'theta1': summary.convergence_metrics.theta1,
            'theta2': summary.convergence_metrics.theta2,
            # Hash ring metrics
            'cache_hits': summary.hash_ring_metrics.local_cache_hits,
            'cache_misses': summary.hash_ring_metrics.local_cache_misses,
            'local_cache_loads': summary.hash_ring_metrics.local_cache_loads,
            'central_storage_loads': summary.hash_ring_metrics.central_storage_loads,
            'cache_write_count': summary.hash_ring_metrics.cache_write_count,
            'cache_hit_rate_percent': (summary.hash_ring_metrics.local_cache_hits / 
                                       (summary.hash_ring_metrics.local_cache_hits + 
                                        summary.hash_ring_metrics.local_cache_misses) * 100.0
                                       if (summary.hash_ring_metrics.local_cache_hits + 
                                           summary.hash_ring_metrics.local_cache_misses) > 0 else 0.0),
            'cache_size_mb': summary.hash_ring_metrics.cache_size_mb,
            'shards_owned': summary.hash_ring_metrics.shards_owned,
            'orphaned_shards': summary.hash_ring_metrics.orphaned_shards,
            'reassigned_shards': summary.hash_ring_metrics.reassigned_shards,
            'recached_shards': summary.hash_ring_metrics.recached_shards,
            'shard_recovery_count': summary.hash_ring_metrics.shard_recovery_count,
            'last_load_source': summary.hash_ring_metrics.last_load_source,
            # Fault metrics
            'injected_faults': summary.fault_metrics.injected_faults,
            'detected_faults': summary.fault_metrics.detected_faults,
            'recovery_attempts': summary.fault_metrics.recovery_attempts,
            'recovered_successfully': summary.fault_metrics.recovered_successfully,
            'recovery_failures': summary.fault_metrics.recovery_failures,
            'total_recovery_time_sec': summary.fault_metrics.total_recovery_time_sec,
            'runtime_fault_triggered': summary.fault_metrics.runtime_fault_triggered,
            'runtime_fault_checkpoint_id': summary.fault_metrics.runtime_fault_checkpoint_id,
            'runtime_fault_epoch': summary.fault_metrics.runtime_fault_epoch,
            'runtime_fault_global_step': summary.fault_metrics.runtime_fault_global_step,
            'resume_attempted': summary.fault_metrics.resume_attempted,
            'resume_succeeded': summary.fault_metrics.resume_succeeded,
            'resumed_from_epoch': summary.fault_metrics.resumed_from_epoch,
            'resumed_checkpoint_id': summary.fault_metrics.resumed_checkpoint_id,
            'resume_load_source': summary.fault_metrics.resume_load_source,
            'time_to_resume_sec': summary.fault_metrics.time_to_resume_sec,
            'rollback_epochs': summary.fault_metrics.rollback_epochs,
            'rollback_steps': summary.fault_metrics.rollback_steps,
            'lost_work_sec': summary.fault_metrics.lost_work_sec,
            # Distributed metrics
            'world_size': summary.distributed_metrics.world_size,
            'num_syncs': summary.distributed_metrics.num_syncs,
            'comm_overhead_percent': summary.distributed_metrics.comm_overhead_percent,
        }
        
        # Append to JSONL file
        with open(output_path, 'a') as f:
            f.write(json.dumps(data) + '\n')
        
        return output_path
    
    # ============================================
    # CSV Export (For Plotting)
    # ============================================
    
    def export_epoch_metrics_csv(self, collector: MetricsCollector,
                                output_path: Optional[str] = None) -> Path:
        """Export per-epoch metrics as CSV for easy plotting.
        
        Args:
            collector: MetricsCollector instance
            output_path: Custom output path (optional)
            
        Returns:
            Path to exported file
        """
        if output_path is None:
            exp_dir = self.base_export_dir / collector.experiment_name
            exp_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = exp_dir / f"{collector.training_mode}_epochs_{timestamp}.csv"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        epoch_metrics = collector.get_epoch_metrics_list()
        if not epoch_metrics:
            return output_path
        
        fieldnames = list(epoch_metrics[0].keys())
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(epoch_metrics)
        
        return output_path
    
    def export_comparison_csv(self, results: List[Dict[str, Any]], 
                             output_path: Optional[str] = None) -> Path:
        """Export comparison results as CSV.
        
        Used to compare multiple training runs side-by-side.
        
        Args:
            results: List of result dictionaries (one per training run)
            output_path: Custom output path (optional)
            
        Returns:
            Path to exported file
        """
        if output_path is None:
            output_path = self.base_export_dir / "comparison_results.csv"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not results:
            return output_path
        
        fieldnames = list(results[0].keys())
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        return output_path
    
    # ============================================
    # HTML Report Generation
    # ============================================
    
    def export_html_report(self, collector: MetricsCollector,
                          output_path: Optional[str] = None) -> Path:
        """Export metrics as an HTML report.
        
        Args:
            collector: MetricsCollector instance
            output_path: Custom output path (optional)
            
        Returns:
            Path to exported file
        """
        summary = collector.finalize()
        
        if output_path is None:
            exp_dir = self.base_export_dir / summary.experiment_name
            exp_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = exp_dir / f"{summary.training_mode}_{timestamp}.html"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        html_content = self._generate_html_report(summary, collector)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path
    
    @staticmethod
    def _generate_html_report(summary: ExperimentSummary, collector: MetricsCollector) -> str:
        """Generate HTML report content."""
        cache_hit_rate = 0.0
        if (summary.hash_ring_metrics.local_cache_hits + 
            summary.hash_ring_metrics.local_cache_misses) > 0:
            cache_hit_rate = (summary.hash_ring_metrics.local_cache_hits / 
                            (summary.hash_ring_metrics.local_cache_hits + 
                             summary.hash_ring_metrics.local_cache_misses)) * 100.0
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Experiment Report: {summary.experiment_name} - {summary.training_mode}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; border-left: 4px solid #007bff; padding-left: 10px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 4px; padding: 15px; }}
        .metric-label {{ font-weight: bold; color: #666; font-size: 0.9em; }}
        .metric-value {{ font-size: 1.5em; color: #007bff; font-weight: bold; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background-color: #007bff; color: white; padding: 10px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .timestamp {{ color: #999; font-size: 0.9em; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Experiment Report: {summary.experiment_name}</h1>
        <p><strong>Training Mode:</strong> {summary.training_mode}</p>
        <p><strong>Dataset:</strong> {summary.dataset} | <strong>Model:</strong> {summary.model}</p>
        <p><strong>Configuration:</strong> {summary.epochs} epochs, batch size {summary.batch_size}</p>
        
        <h2>Training Performance</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Best Validation Accuracy</div>
                <div class="metric-value">{summary.best_val_accuracy:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Final Validation Loss</div>
                <div class="metric-value">{summary.final_val_loss:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Best Training Loss</div>
                <div class="metric-value">{summary.best_train_loss:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Training Time</div>
                <div class="metric-value">{summary.total_training_time_sec:.1f}s</div>
            </div>
        </div>
        
        <h2>Checkpointing Overhead</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Total Checkpoint Count</div>
                <div class="metric-value">{summary.total_checkpoint_count}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Checkpoint Time</div>
                <div class="metric-value">{summary.total_checkpoint_time_sec:.1f}s</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Checkpoint Overhead</div>
                <div class="metric-value">{summary.checkpoint_overhead_percent:.1f}%</div>
            </div>
        </div>
        
        <h2>Convergence Scheduler Metrics</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Current Interval</div>
                <div class="metric-value">{summary.convergence_metrics.current_interval_sec:.2f}s</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Convergence Checkpoints</div>
                <div class="metric-value">{summary.convergence_metrics.total_convergence_checkpoints}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Theta1 (Decay)</div>
                <div class="metric-value">{summary.convergence_metrics.theta1 or 'N/A'}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Theta2 (Offset)</div>
                <div class="metric-value">{summary.convergence_metrics.theta2 or 'N/A'}</div>
            </div>
        </div>
        
        <h2>Hash Ring Metrics</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Cache Hits</div>
                <div class="metric-value">{summary.hash_ring_metrics.local_cache_hits}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Cache Misses</div>
                <div class="metric-value">{summary.hash_ring_metrics.local_cache_misses}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Cache Hit Rate</div>
                <div class="metric-value">{cache_hit_rate:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Cache Size</div>
                <div class="metric-value">{summary.hash_ring_metrics.cache_size_mb:.1f}MB</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Shards Owned</div>
                <div class="metric-value">{summary.hash_ring_metrics.shards_owned}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Central Storage Loads</div>
                <div class="metric-value">{summary.hash_ring_metrics.central_storage_loads}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Re-cached Shards</div>
                <div class="metric-value">{summary.hash_ring_metrics.recached_shards}</div>
            </div>
        </div>
        
        <h2>Fault Tolerance Metrics</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Injected Faults</div>
                <div class="metric-value">{summary.fault_metrics.injected_faults}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Recovered Successfully</div>
                <div class="metric-value">{summary.fault_metrics.recovered_successfully}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Recovery Failures</div>
                <div class="metric-value">{summary.fault_metrics.recovery_failures}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Recovery Time</div>
                <div class="metric-value">{summary.fault_metrics.total_recovery_time_sec:.1f}s</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Resume Time</div>
                <div class="metric-value">{summary.fault_metrics.time_to_resume_sec:.1f}s</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Rollback Epochs</div>
                <div class="metric-value">{summary.fault_metrics.rollback_epochs:.2f}</div>
            </div>
        </div>
        
        <h2>Distributed Training Metrics</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">World Size</div>
                <div class="metric-value">{summary.distributed_metrics.world_size}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Synchronizations</div>
                <div class="metric-value">{summary.distributed_metrics.num_syncs}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Communication Overhead</div>
                <div class="metric-value">{summary.distributed_metrics.comm_overhead_percent:.1f}%</div>
            </div>
        </div>
        
        <p class="timestamp">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
</body>
</html>
"""
        return html
    
    # ============================================
    # Markdown Report Generation
    # ============================================
    
    def export_markdown_report(self, collector: MetricsCollector,
                              output_path: Optional[str] = None) -> Path:
        """Export metrics as a Markdown report.
        
        Args:
            collector: MetricsCollector instance
            output_path: Custom output path (optional)
            
        Returns:
            Path to exported file
        """
        summary = collector.finalize()
        
        if output_path is None:
            exp_dir = self.base_export_dir / summary.experiment_name
            exp_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = exp_dir / f"{summary.training_mode}_{timestamp}.md"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        md = f"""# Experiment Report: {summary.experiment_name}

**Training Mode:** {summary.training_mode}  
**Dataset:** {summary.dataset}  
**Model:** {summary.model}  
**Configuration:** {summary.epochs} epochs, batch size {summary.batch_size}  
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Training Performance

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | {summary.best_val_accuracy:.2f}% |
| Final Validation Accuracy | {summary.final_val_accuracy:.2f}% |
| Best Validation Loss | {summary.best_val_loss:.4f} |
| Final Validation Loss | {summary.final_val_loss:.4f} |
| Best Training Loss | {summary.best_train_loss:.4f} |
| Final Training Loss | {summary.final_train_loss:.4f} |
| Total Training Time | {summary.total_training_time_sec:.1f}s |

## Checkpointing Overhead

| Metric | Value |
|--------|-------|
| Total Checkpoint Count | {summary.total_checkpoint_count} |
| Total Checkpoint Time | {summary.total_checkpoint_time_sec:.1f}s |
| Checkpoint Overhead | {summary.checkpoint_overhead_percent:.1f}% |

## Convergence Scheduler Metrics

| Metric | Value |
|--------|-------|
| Current Interval (seconds) | {summary.convergence_metrics.current_interval_sec:.2f} |
| Total Convergence Checkpoints | {summary.convergence_metrics.total_convergence_checkpoints} |
| Loss Fit Segments Restarted | {summary.convergence_metrics.segments_since_restart} |
| Theta1 (Exponential Decay) | {summary.convergence_metrics.theta1 or 'N/A'} |
| Theta2 (Exponential Offset) | {summary.convergence_metrics.theta2 or 'N/A'} |

## Hash Ring Metrics

| Metric | Value |
|--------|-------|
| Local Cache Hits | {summary.hash_ring_metrics.local_cache_hits} |
| Local Cache Misses | {summary.hash_ring_metrics.local_cache_misses} |
| Cache Size (MB) | {summary.hash_ring_metrics.cache_size_mb:.1f} |
| Max Cache Size (MB) | {summary.hash_ring_metrics.max_cache_size_mb:.1f} |
| Total Shards | {summary.hash_ring_metrics.total_shards} |
| Shards Owned by This Node | {summary.hash_ring_metrics.shards_owned} |
| Central Storage Loads | {summary.hash_ring_metrics.central_storage_loads} |
| Local Cache Loads | {summary.hash_ring_metrics.local_cache_loads} |
| Cache Writes | {summary.hash_ring_metrics.cache_write_count} |
| Orphaned Shards | {summary.hash_ring_metrics.orphaned_shards} |
| Reassigned Shards | {summary.hash_ring_metrics.reassigned_shards} |
| Re-cached Shards | {summary.hash_ring_metrics.recached_shards} |
| Shard Recovery Count | {summary.hash_ring_metrics.shard_recovery_count} |
| Total Recovery Time | {summary.hash_ring_metrics.recovery_time_sec:.1f}s |
| Last Load Source | {summary.hash_ring_metrics.last_load_source or 'N/A'} |

## Fault Tolerance Metrics

| Metric | Value |
|--------|-------|
| Injected Faults | {summary.fault_metrics.injected_faults} |
| Detected Faults | {summary.fault_metrics.detected_faults} |
| Recovery Attempts | {summary.fault_metrics.recovery_attempts} |
| Successfully Recovered | {summary.fault_metrics.recovered_successfully} |
| Recovery Failures | {summary.fault_metrics.recovery_failures} |
| Checkpoint Integrity Failures | {summary.fault_metrics.checkpoint_integrity_failures} |
| Total Recovery Time | {summary.fault_metrics.total_recovery_time_sec:.1f}s |
| Runtime Fault Triggered | {summary.fault_metrics.runtime_fault_triggered} |
| Runtime Fault Checkpoint | {summary.fault_metrics.runtime_fault_checkpoint_id or 'N/A'} |
| Runtime Fault Epoch | {summary.fault_metrics.runtime_fault_epoch if summary.fault_metrics.runtime_fault_epoch is not None else 'N/A'} |
| Runtime Fault Global Step | {summary.fault_metrics.runtime_fault_global_step if summary.fault_metrics.runtime_fault_global_step is not None else 'N/A'} |
| Resume Attempted | {summary.fault_metrics.resume_attempted} |
| Resume Succeeded | {summary.fault_metrics.resume_succeeded} |
| Resumed From Epoch | {summary.fault_metrics.resumed_from_epoch if summary.fault_metrics.resumed_from_epoch is not None else 'N/A'} |
| Resumed Checkpoint | {summary.fault_metrics.resumed_checkpoint_id or 'N/A'} |
| Resume Load Source | {summary.fault_metrics.resume_load_source or 'N/A'} |
| Time to Resume | {summary.fault_metrics.time_to_resume_sec:.1f}s |
| Rollback Epochs | {summary.fault_metrics.rollback_epochs:.2f} |
| Rollback Steps | {summary.fault_metrics.rollback_steps} |
| Lost Work Time | {summary.fault_metrics.lost_work_sec:.1f}s |

## Distributed Training Metrics

| Metric | Value |
|--------|-------|
| World Size | {summary.distributed_metrics.world_size} |
| Total Synchronization Operations | {summary.distributed_metrics.num_syncs} |
| Total Synchronization Time | {summary.distributed_metrics.total_sync_time_sec:.1f}s |
| Communication Overhead | {summary.distributed_metrics.comm_overhead_percent:.1f}% |
"""
        
        with open(output_path, 'w') as f:
            f.write(md)
        
        return output_path
    
    # ============================================
    # Utility Methods
    # ============================================
    
    @staticmethod
    def load_jsonl_results(file_path: str) -> List[Dict[str, Any]]:
        """Load results from JSONL file.
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            List of result dictionaries
        """
        results = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        return results
    
    @staticmethod
    def compare_modes(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comparison statistics across different training modes.
        
        Args:
            results: List of result dictionaries from JSONL
            
        Returns:
            Comparison statistics
        """
        if not results:
            return {}
        
        by_mode = {}
        for result in results:
            mode = result.get('training_mode', 'unknown')
            if mode not in by_mode:
                by_mode[mode] = []
            by_mode[mode].append(result)
        
        comparison = {}
        for mode, mode_results in by_mode.items():
            n = len(mode_results)
            if n == 0:
                continue
            
            avg_training_time = sum(r.get('total_training_time_sec', 0) for r in mode_results) / n
            avg_val_acc = sum(r.get('best_val_accuracy', 0) for r in mode_results) / n
            avg_checkpoint_overhead = sum(r.get('checkpoint_overhead_percent', 0) for r in mode_results) / n
            avg_cache_hit_rate = sum(r.get('cache_hit_rate_percent', 0) for r in mode_results) / n
            
            comparison[mode] = {
                'num_runs': n,
                'avg_training_time_sec': avg_training_time,
                'avg_best_val_accuracy': avg_val_acc,
                'avg_checkpoint_overhead_percent': avg_checkpoint_overhead,
                'avg_cache_hit_rate_percent': avg_cache_hit_rate,
            }
        
        return comparison
