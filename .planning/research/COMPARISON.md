# Comparison: Fault Tolerance Approaches

**Context:** Adding fault injection and recovery to PyTorch DDP multi-GPU training

## Quick Comparison

| Criterion | Checkpoint-based (v1.1) | torchft (Production) |
|-----------|-------------------------|---------------------|
| **Complexity** | Low | High |
| **Dependencies** | None (existing) | torchft library |
| **Recovery Time** | Minutes (restart) | Seconds (peer transfer) |
| **Implementation** | Exception handling | Full library integration |
| **Testing Suitability** | Ideal | Overkill for testing |

## Recommendation

**Use Checkpoint-based (v1.1)** for this project's requirements because:
- Test/verification use case, not production training
- Existing CheckpointManager supports DDP
- No new dependencies required
- Simpler to implement and maintain

**Use torchft** when:
- Live recovery needed (no restart)
- Production LLM training at scale
- Complex multi-replica coordination

## Sources
- Meta torchft: https://github.com/meta-pytorch/torchft
- PyTorch fault tolerant tutorial: https://pytorch.org/tutorials/beginner/ddp_series_fault_tolerance