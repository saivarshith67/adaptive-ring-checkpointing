#!/usr/bin/env python3
"""
Example script showing how to integrate metrics collection into training.

This demonstrates:
1. Creating a MetricsCollector
2. Recording metrics during training
3. Exporting in multiple formats
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.coci.metrics import MetricsCollector, MetricsExporter


def example_training_with_metrics():
    """Example of training loop with metrics collection."""
    
    # ============================================================
    # Step 1: Initialize Metrics Collector
    # ============================================================
    metrics = MetricsCollector(
        experiment_name="example_experiment",
        training_mode="convergence-hash-ring",
        dataset="faceforensics",
        model="efficientnet-b0",
        epochs=10,
        batch_size=32
    )
    
    print("✓ Metrics collector initialized")
    
    # ============================================================
    # Step 2: Simulate Training Loop
    # ============================================================
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Mark epoch start
        metrics.start_epoch(epoch)
        
        # Simulate training
        train_loss = 1.0 - (epoch * 0.15)  # Simulated decreasing loss
        train_acc = 50.0 + (epoch * 8.0)   # Simulated increasing accuracy
        
        # Simulate validation
        val_loss = 0.95 - (epoch * 0.12)
        val_acc = 55.0 + (epoch * 7.5)
        
        # ============================================================
        # Step 3: Record Epoch Metrics
        # ============================================================
        metrics.end_epoch(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc
        )
        
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.1f}%")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.1f}%")
        
        # ============================================================
        # Step 4: Record Checkpoint Metrics
        # ============================================================
        if (epoch + 1) % 2 == 0:  # Checkpoint every 2 epochs
            checkpoint_size = 120.5 + (epoch * 1.2)  # MB
            save_time = 2.5 + (epoch * 0.1)  # seconds
            
            metrics.record_checkpoint(
                checkpoint_size_mb=checkpoint_size,
                save_time_sec=save_time,
                checkpoint_id=f"epoch_{epoch:03d}"
            )
            print(f"  ✓ Checkpoint: {checkpoint_size:.1f}MB in {save_time:.2f}s")
        
        # ============================================================
        # Step 5: Record Convergence Metrics (if using COCI)
        # ============================================================
        interval = 300.0 - (epoch * 30.0)  # Interval decreases as converging
        metrics.update_convergence_metrics(
            current_interval_sec=interval,
            theta1=-0.002 - (epoch * 0.0001),
            theta2=0.5 - (epoch * 0.03),
            loss_fit_error=0.02 + (epoch * 0.001)
        )
        
        # Increment convergence checkpoint counter
        metrics.increment_convergence_checkpoint()
        
        # ============================================================
        # Step 6: Record Hash Ring Metrics (if using hash-ring)
        # ============================================================
        metrics.update_hash_ring_metrics(
            total_shards=8,
            cache_size_mb=200.0 + (epoch * 50.0),
            max_cache_mb=1024.0,
            shards_owned=2
        )
        
        # Simulate cache operations
        if epoch % 2 == 0:
            metrics.increment_cache_hit()
            metrics.increment_cache_hit()
            metrics.increment_cache_miss()
        
        # ============================================================
        # Step 7: Record Distributed Training Metrics
        # ============================================================
        metrics.set_distributed_config(world_size=4)
        metrics.record_synchronization(sync_time_sec=0.5)
        
        # Simulate batch processing
        for batch in range(100):
            metrics.increment_batch_count(samples=32)
        
        time.sleep(0.1)  # Simulate training time
    
    # ============================================================
    # Step 8: Finalize and Export
    # ============================================================
    print("\n" + "=" * 60)
    print("Training Complete! Exporting Metrics...")
    print("=" * 60)
    
    # Create exporter
    exporter = MetricsExporter(base_export_dir="./experiment_results")
    
    # Export in multiple formats
    json_path = exporter.export_summary_json(metrics)
    print(f"✓ JSON summary: {json_path}")
    
    jsonl_path = exporter.export_summary_jsonl(metrics)
    print(f"✓ JSONL appended to: {jsonl_path}")
    
    csv_path = exporter.export_epoch_metrics_csv(metrics)
    print(f"✓ Epoch CSV: {csv_path}")
    
    html_path = exporter.export_html_report(metrics)
    print(f"✓ HTML report: {html_path}")
    
    md_path = exporter.export_markdown_report(metrics)
    print(f"✓ Markdown report: {md_path}")
    
    # ============================================================
    # Step 9: Print Summary
    # ============================================================
    summary = metrics.finalize()
    print("\n" + "=" * 60)
    print("SUMMARY METRICS")
    print("=" * 60)
    print(f"Total Training Time: {summary.total_training_time_sec:.1f}s")
    print(f"Total Checkpoints: {summary.total_checkpoint_count}")
    print(f"Checkpoint Overhead: {summary.checkpoint_overhead_percent:.2f}%")
    print(f"Best Val Accuracy: {summary.best_val_accuracy:.2f}%")
    print(f"Convergence Checkpoints: {summary.convergence_metrics.total_convergence_checkpoints}")
    print(f"Cache Hit Rate: {(summary.hash_ring_metrics.local_cache_hits / (summary.hash_ring_metrics.local_cache_hits + summary.hash_ring_metrics.local_cache_misses) * 100) if (summary.hash_ring_metrics.local_cache_hits + summary.hash_ring_metrics.local_cache_misses) > 0 else 0:.1f}%")
    print(f"World Size: {summary.distributed_metrics.world_size}")
    
    return 0


def example_compare_results():
    """Example of comparing results across modes."""
    print("\n" + "=" * 60)
    print("COMPARING RESULTS")
    print("=" * 60)
    
    exporter = MetricsExporter()
    
    try:
        # Load all results
        results = exporter.load_jsonl_results("./experiment_results/experiment_summaries.jsonl")
        print(f"Loaded {len(results)} experiment results\n")
        
        # Compare modes
        comparison = exporter.compare_modes(results)
        
        if comparison:
            print("Mode Comparison:")
            print(f"{'Mode':<25} {'Avg Time (s)':<15} {'Avg Overhead %':<15} {'Avg Val Acc %':<15}")
            print("-" * 70)
            
            for mode, stats in comparison.items():
                print(f"{mode:<25} {stats['avg_training_time_sec']:<15.1f} {stats['avg_checkpoint_overhead_percent']:<15.2f} {stats['avg_best_val_accuracy']:<15.2f}")
        else:
            print("No results to compare yet. Run training first.")
            
    except FileNotFoundError:
        print("No results found. Run training first with metrics collection.")


if __name__ == "__main__":
    # Run example training with metrics
    result = example_training_with_metrics()
    
    # Show comparison (if multiple runs exist)
    example_compare_results()
    
    sys.exit(result)
