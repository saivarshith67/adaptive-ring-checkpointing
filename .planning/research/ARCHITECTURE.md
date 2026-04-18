# Architecture Patterns: Multi-GPU Fault Injection and Recovery

**Domain:** Fault tolerance for distributed PyTorch DDP training
**Researched:** 2026-04-17

## Recommended Architecture

Three-component model:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training Loop + Recovery                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────┐      │
│  │  Train   │───▶│  Fault   │───▶│  CheckpointManager   │      │
│  │  Epoch   │    │  Inject  │    │  (DDP-aware save)    │      │
│  └──────────┘    └──────────┘    └──────────────────────┘      │
│       │                                           ▲             │
│       │                                           │             │
│       ▼                                           │             │
│  ┌──────────┐    ┌──────────┐                      │             │
│  │  Except  │───▶│ Barrier  │──────────────────────┘             │
│  │  Handler │    │  Sync    │                                     │
│  └──────────┘    └──────────┘                                     │
│                                                                  │
│  Recovery on Restart:                                            │
│  ┌──────────────────────┐                                        │
│  │  CheckpointManager   │───▶ Resume training from epoch         │
│  │  .load_latest()      │                                        │
│  └──────────────────────┘                                        │
└─────────────────────────────────────────────────────────────────┘
```

## Integration with Existing Code

### Modify: FaultInjector
```python
class FaultInjector:
    def __init__(self, rank: int = None, failure_probability: float = 0.1, ...):
        self.rank = rank
        self.failure_probability = failure_probability
        
    def maybe_fail(self):
        # Check rank if specified
        if self.rank is not None:
            if torch.distributed.get_rank() != self.rank:
                return  # Skip on other ranks
        
        # Existing failure logic
        if random.random() < self.failure_probability:
            raise RuntimeError(f"Injected fault on rank {torch.distributed.get_rank()}")
```

### Modify: Training Loop
```python
def train_with_fault_tolerance(model, train_loader, optimizer, ...):
    for epoch in range(start_epoch, epochs):
        try:
            avg_loss = train_epoch(model, train_loader, optimizer, epoch)
            
            # Fault injection after epoch
            if fault_injector:
                fault_injector.maybe_fail()
                
        except RuntimeError as e:
            logger.warning(f"Fault detected: {e}")
            
            # Sync all ranks
            if is_distributed_initialized():
                dist.barrier()
            
            # Emergency checkpoint save
            checkpoint_manager.save(model, optimizer, epoch, avg_loss)
            
            raise  # Let torchrun handle restart
```

### Recovery: Uses Existing
```python
# CheckpointManager.load_latest() already handles:
# - DDP unwrapping (model.module.load_state_dict)
# - Epoch tracking
# - Optimizer state
start_epoch = checkpoint_manager.load_latest(model, optimizer, device)
```

## Checkpoint Contents for Recovery

A complete snapshot must include:
```python
snapshot = {
    'model': model.module.state_dict(),      # Unwrap DDP
    'optimizer': optimizer.state_dict(),      # Continue gradient descent
    'scheduler': scheduler.state_dict() if scheduler else None,
    'epoch': current_epoch,                  # Resume from next epoch
    'rng_cpu': torch.get_rng_state(),        # Reproducible data
    'rng_gpu': torch.cuda.get_rng_state_all(),  # GPU randomness
    'sampler_state': sampler.state_dict(),   # Data loader position
}
```

## Sources

- [PyTorch Fault-Tolerant DDP Tutorial](https://pytorch.org/tutorials/beginner/ddp_series_fault_tolerance) — HIGH confidence
- [torchrun Documentation](https://docs.pytorch.org/docs/stable/elastic/run.html) — HIGH confidence
