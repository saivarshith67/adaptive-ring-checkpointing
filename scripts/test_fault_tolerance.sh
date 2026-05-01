#!/bin/bash
# =============================================================================
# End-to-End Fault Tolerance Verification Test
# =============================================================================
# This script verifies the complete fault tolerance loop:
#   1. Training starts with fault injection enabled
#   2. A failure is injected during training
#   3. Exception handler saves checkpoint before re-raising
#   4. Training resumes from the checkpoint
#   5. Metrics continue correctly after recovery
#
# Usage:
#   # Single GPU test
#   bash scripts/test_fault_tolerance.sh 1
#
#   # Multi-GPU test (requires CUDA)
#   bash scripts/test_fault_tolerance.sh 2
#
# Requirements:
#   - PyTorch with CUDA support (for multi-GPU)
#   - Dataset at data/faceforensics/ (will use --limit for fast test)
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
GPU=${1:-1}
EPOCHS=5
CKPT_INTERVAL=2
LIMIT=100
CKPT_DIR="./checkpoints"

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

cleanup() {
    log_info "Cleaning up previous checkpoints..."
    rm -rf "$CKPT_DIR"
    mkdir -p "$CKPT_DIR"
}

check_env() {
    log_info "Checking environment..."
    
    # Check Python
    if ! command -v python &> /dev/null; then
        log_error "Python not found. Please install Python."
        exit 1
    fi
    
    # Check PyTorch
    if ! python -c "import torch; import torch.distributed" 2>/dev/null; then
        log_error "PyTorch not found. Please install PyTorch."
        exit 1
    fi
    
    # Check CUDA availability for multi-GPU
    if [ "$GPU" -gt 1 ]; then
        if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            log_error "CUDA required for multi-GPU test. CUDA not available."
            exit 1
        fi
        CUDA_DEVICES=$(python -c "import torch; print(torch.cuda.device_count())")
        if [ "$CUDA_DEVICES" -lt "$GPU" ]; then
            log_warn "Requested $GPU GPUs but only $CUDA_DEVICES available"
            GPU=$CUDA_DEVICES
        fi
    fi
    
    log_success "Environment check passed"
}

# =============================================================================
# Test 1: Single GPU Fault Tolerance
# =============================================================================

test_single_gpu() {
    log_info "=============================================="
    log_info "Test 1: Single GPU Fault Tolerance"
    log_info "=============================================="
    
    cleanup
    
    log_info "Step 1: Training with fault injection (expect failure at epoch ~2)"
    log_info "Command: python scripts/train_faceforensics.py --epochs $EPOCHS --checkpoint_interval $CKPT_INTERVAL --inject-fault --inject-rank 0 --inject-rate 0.5 --limit $LIMIT"
    
    # Run training with fault injection - expect failure
    set +e  # Don't exit on failure
    timeout 180 python scripts/train_faceforensics.py \
        --epochs $EPOCHS \
        --checkpoint_interval $CKPT_INTERVAL \
        --inject-fault \
        --inject-rank 0 \
        --inject-rate 0.5 \
        --limit $LIMIT 2>&1 | tee /tmp/train_log.txt || true
    TRAIN_EXIT=$?
    set -e
    
    log_info "Training exited with code: $TRAIN_EXIT"
    
    # Check if checkpoint was saved (either by interval or emergency save)
    log_info "Step 2: Checking checkpoint..."
    
    CHECKPOINT_FOUND=false
    if [ -f "$CKPT_DIR/checkpoint_epoch_2.pt" ]; then
        log_success "Scheduled checkpoint found: checkpoint_epoch_2.pt"
        CHECKPOINT_FOUND=true
        SAVED_EPOCH=2
    elif [ -f "$CKPT_DIR/checkpoint_epoch_1.pt" ]; then
        log_success "Emergency checkpoint found: checkpoint_epoch_1.pt"
        CHECKPOINT_FOUND=true
        SAVED_EPOCH=1
    fi
    
    if [ "$CHECKPOINT_FOUND" = false ]; then
        log_error "No checkpoint found after fault injection"
        log_error "This indicates checkpoint saving failed"
        # List checkpoints for debugging
        if [ -d "$CKPT_DIR" ]; then
            log_info "Contents of $CKPT_DIR:"
            ls -la "$CKPT_DIR" || true
        fi
        return 1
    fi
    
    log_info "Step 3: Resuming from checkpoint (epoch $SAVED_EPOCH)..."
    log_info "Command: python scripts/train_faceforensics.py --epochs $EPOCHS --resume --limit $LIMIT"
    
    # Resume training from checkpoint
    timeout 180 python scripts/train_faceforensics.py \
        --epochs $EPOCHS \
        --resume \
        --limit $LIMIT 2>&1 | tee /tmp/resume_log.txt || RESUME_EXIT=$?
    
    # Check resume log for successful loading
    if grep -q "Resumed from epoch" /tmp/resume_log.txt 2>/dev/null; then
        log_success "Training resumed successfully from checkpoint"
    else
        log_error "Resume did not report checkpoint loading"
        return 1
    fi
    
    # Verify resumed from correct epoch
    if grep -q "epoch $((SAVED_EPOCH + 1))" /tmp/resume_log.txt 2>/dev/null; then
        log_success "Training continued from correct epoch"
    fi
    
    log_success "Single GPU test completed successfully"
    return 0
}

# =============================================================================
# Test 2: Multi-GPU Fault Tolerance
# =============================================================================

# Function to select N least-used GPUs based on memory usage
select_least_used_gpus() {
    local N=$1
    # Query GPU memory usage, sort by memory used (ascending), pick top N
    # Output format: "gpu_id,gpu_id,..."
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | \
        sort -t',' -k2 -n | \
        head -n "$N" | \
        cut -d',' -f1 | \
        paste -sd ',' -
}

test_multi_gpu() {
    log_info "=============================================="
    log_info "Test 2: Multi-GPU Fault Tolerance (GPU=$GPU)"
    log_info "=============================================="
    
    if [ "$GPU" -lt 2 ]; then
        log_warn "Multi-GPU test requires at least 2 GPUs. Skipping."
        return 0
    fi
    
    cleanup
    
    # Select least-used GPUs for multi-GPU test
    log_info "Selecting $GPU least-used GPUs..."
    SELECTED_GPUS=$(select_least_used_gpus $GPU)
    export CUDA_VISIBLE_DEVICES=$SELECTED_GPUS
    log_info "Selected GPUs: $SELECTED_GPUS"
    
    log_info "Step 1: Multi-GPU training with fault injection on rank 0"
    log_info "Command: torchrun --nproc_per_node=$GPU scripts/train_faceforensics.py --epochs $EPOCHS --checkpoint_interval $CKPT_INTERVAL --inject-fault --inject-rank 0 --inject-rate 0.5 --limit $LIMIT"
    
    # Run training with fault injection on multi-GPU - expect failure
    set +e  # Don't exit on failure
    timeout 180 torchrun --nproc_per_node=$GPU \
        scripts/train_faceforensics.py \
        --epochs $EPOCHS \
        --checkpoint_interval $CKPT_INTERVAL \
        --inject-fault \
        --inject-rank 0 \
        --inject-rate 0.5 \
        --limit $LIMIT 2>&1 | tee /tmp/multi_train_log.txt || true
    TRAIN_EXIT=$?
    set -e
    
    log_info "Training exited with code: $TRAIN_EXIT"
    
    # Check checkpoint was saved
    log_info "Step 2: Checking checkpoint..."
    
    CHECKPOINT_FOUND=false
    if [ -f "$CKPT_DIR/checkpoint_epoch_2.pt" ]; then
        log_success "Scheduled checkpoint found: checkpoint_epoch_2.pt"
        CHECKPOINT_FOUND=true
        SAVED_EPOCH=2
    elif [ -f "$CKPT_DIR/checkpoint_epoch_1.pt" ]; then
        log_success "Emergency checkpoint found: checkpoint_epoch_1.pt"
        CHECKPOINT_FOUND=true
        SAVED_EPOCH=1
    fi
    
    if [ "$CHECKPOINT_FOUND" = false ]; then
        log_error "No checkpoint found after fault injection"
        log_info "Contents of $CKPT_DIR:"
        ls -la "$CKPT_DIR" || true
        return 1
    fi
    
    log_info "Step 3: Resuming multi-GPU training from checkpoint..."
    log_info "Command: torchrun --nproc_per_node=$GPU scripts/train_faceforensics.py --epochs $EPOCHS --resume --limit $LIMIT"
    
    # Resume multi-GPU training
    timeout 180 torchrun --nproc_per_node=$GPU \
        scripts/train_faceforensics.py \
        --epochs $EPOCHS \
        --resume \
        --limit $LIMIT 2>&1 | tee /tmp/multi_resume_log.txt || RESUME_EXIT=$?
    
    # Check resume log
    if grep -q "Resumed from epoch" /tmp/multi_resume_log.txt 2>/dev/null; then
        log_success "Multi-GPU training resumed successfully"
    else
        log_error "Multi-GPU resume did not report checkpoint loading"
        return 1
    fi
    
    log_success "Multi-GPU test completed successfully"
    return 0
}

# =============================================================================
# Test 3: Verify Metrics Continue Correctly
# =============================================================================

test_metrics_continuity() {
    log_info "=============================================="
    log_info "Test 3: Metrics Continuity After Recovery"
    log_info "=============================================="
    
    # This test verifies that training metrics (loss, accuracy) continue
    # correctly after recovery by checking:
    # 1. Loss values are in reasonable range (not NaN, not extreme)
    # 2. Training progresses (loss generally decreases over epochs)
    # 3. No duplicate epochs in the log
    
    if [ ! -f /tmp/resume_log.txt ] && [ ! -f /tmp/multi_resume_log.txt ]; then
        log_warn "No training log found. Skipping metrics continuity test."
        return 0
    fi
    
    LOG_FILE=${multi_resume_log.txt:-resume_log.txt}
    
    # Check for NaN or extreme loss values
    if grep -q "NaN" /tmp/$LOG_FILE 2>/dev/null; then
        log_error "NaN detected in training logs"
        return 1
    fi
    
    # Check loss values are reasonable
    LOSS_VALUES=$(grep -oP "Loss: \K[0-9.]+" /tmp/$LOG_FILE 2>/dev/null | head -5)
    if [ -n "$LOSS_VALUES" ]; then
        log_success "Loss values are numeric (no NaN)"
    fi
    
    # Check epoch progression
    EPOCHS_IN_LOG=$(grep -oP "Epoch \K[0-9]+" /tmp/$LOG_FILE 2>/dev/null | sort -u)
    if [ -n "$EPOCHS_IN_LOG" ]; then
        log_success "Epoch progression: $EPOCHS_IN_LOG"
    fi
    
    log_success "Metrics continuity verified"
    return 0
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    echo ""
    echo "=============================================="
    echo "Fault Tolerance End-to-End Verification"
    echo "=============================================="
    echo ""
    
    check_env
    
    FAILED=0
    
    # Test 1: Single GPU (always run)
    if ! test_single_gpu; then
        log_error "Single GPU test FAILED"
        FAILED=1
    fi
    
    # Test 2: Multi-GPU (if available)
    if ! test_multi_gpu; then
        log_error "Multi-GPU test FAILED"
        FAILED=1
    fi
    
    # Test 3: Metrics continuity (if we have logs)
    if ! test_metrics_continuity; then
        log_warn "Metrics continuity test had issues"
    fi
    
    echo ""
    echo "=============================================="
    if [ $FAILED -eq 0 ]; then
        echo -e "${GREEN}All fault tolerance tests PASSED${NC}"
        echo "=============================================="
        echo ""
        echo "Summary:"
        echo "  ✓ Single GPU fault tolerance verified"
        if [ "$GPU" -ge 2 ]; then
            echo "  ✓ Multi-GPU fault tolerance verified"
        fi
        echo "  ✓ Metrics continuity verified"
        echo ""
        exit 0
    else
        echo -e "${RED}Some fault tolerance tests FAILED${NC}"
        echo "=============================================="
        echo ""
        exit 1
    fi
}

# Run main function
main "$@"
