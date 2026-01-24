#!/bin/bash
# Filename: fix_tensorrt_llm_deps.sh
# Location: phase-1-foundation/week-02-tensorrt/
# Purpose: Fix TensorRT-LLM dependencies (MPI library)

echo "========================================"
echo "FIXING TENSORRT-LLM DEPENDENCIES"
echo "========================================"

# Install OpenMPI (provides libmpi.so.40)
echo ""
echo "[1/3] Installing OpenMPI..."
sudo apt-get update
sudo apt-get install -y openmpi-bin libopenmpi-dev

# Verify MPI installation
echo ""
echo "[2/3] Verifying MPI installation..."
ls -la /usr/lib/x86_64-linux-gnu/libmpi.so* 2>/dev/null || \
ls -la /usr/lib/libmpi.so* 2>/dev/null || \
echo "Checking alternative locations..."

# Find and display libmpi location
echo ""
echo "MPI library locations:"
find /usr -name "libmpi.so*" 2>/dev/null

# Test TensorRT-LLM import
echo ""
echo "[3/3] Testing TensorRT-LLM import..."
source ~/rtx3090-ai-training/ai-inference/bin/activate

python3 -c "
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorrt_llm
    print(f'✓ TensorRT-LLM version: {tensorrt_llm.__version__}')
    print(f'✓ Import successful!')
except ImportError as e:
    print(f'✗ Import failed: {e}')
except Exception as e:
    print(f'✗ Other error: {e}')
"

echo ""
echo "========================================"
echo "DEPENDENCY FIX COMPLETE"
echo "========================================"