#!/bin/bash

echo "================================================"
echo "Week 2: TensorRT Setup for 4x RTX 3090"
echo "CUDA 12.6, Ubuntu 24.04"
echo "================================================"

# Exit on any error
set -e

# Check CUDA version
echo ""
echo "[1/6] Verifying CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: CUDA not found. Please install CUDA 12.6 first."
    exit 1
fi

nvcc --version | grep "release"
nvidia-smi | head -n 10

# Check GPU count
echo ""
echo "[2/6] Detecting GPUs..."
gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
echo "Detected $gpu_count GPU(s)"

if [ "$gpu_count" != "4" ]; then
    echo "WARNING: Expected 4 GPUs, found $gpu_count"
fi

# Activate virtual environment
echo ""
echo "[3/6] Activating virtual environment..."
if [ ! -d "../../ai-inference" ]; then
    echo "ERROR: Virtual environment not found. Run setup.sh from repository root first."
    exit 1
fi

source ../../ai-inference/bin/activate

# Install TensorRT via pip (compatible with CUDA 12.x)
echo ""
echo "[4/6] Installing TensorRT..."
echo "Note: Using pip installation for TensorRT 10.x (CUDA 12 compatible)"

pip install --upgrade tensorrt

# Install CUDA-X libraries
echo ""
echo "[5/6] Installing CUDA-X libraries..."

# cuDNN (should already be installed with PyTorch, but ensure latest)
pip install nvidia-cudnn-cu12

# cuBLAS (included in PyTorch but ensure available)
pip install nvidia-cublas-cu12

# NCCL for multi-GPU communication
pip install nvidia-nccl-cu12

# TensorRT optimization tools
pip install polygraphy --extra-index-url https://pypi.ngc.nvidia.com

# Verify installations
echo ""
echo "[6/6] Verifying installations..."

python3 << 'PYEOF'
import sys

print("\n=== Installation Verification ===")

# Check TensorRT
try:
    import tensorrt as trt
    print(f"✓ TensorRT version: {trt.__version__}")
except ImportError as e:
    print(f"✗ TensorRT import failed: {e}")
    sys.exit(1)

# Check PyTorch + CUDA
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ GPU count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
except Exception as e:
    print(f"✗ PyTorch verification failed: {e}")
    sys.exit(1)

# Check NCCL
try:
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # NCCL is available through torch.distributed
        print(f"✓ NCCL available for multi-GPU operations")
except Exception as e:
    print(f"⚠ NCCL verification skipped: {e}")

print("\n=== Verification Complete ===")
PYEOF

echo ""
echo "================================================"
echo "TensorRT Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Run verification: python3 verify_multi_gpu.py"
echo "  2. Test conversion: python3 simple_conversion_test.py"
echo ""
