#!/bin/bash
# Filename: benchmark_llama1b_trt_fixed.sh
# Location: phase-1-foundation/week-02-tensorrt/
# Purpose: Run benchmark with TensorRT libraries in LD_LIBRARY_PATH

echo "========================================"
echo "TENSORRT BENCHMARK WITH FIXED LIB PATH"
echo "========================================"

cd ~/work/rtx3090-ai-training/phase-1-foundation/week-02-tensorrt
. ~/ai-inference/bin/activate

# Add TensorRT libraries to path
export LD_LIBRARY_PATH="/home/msamuels/ai-inference/lib/python3.12/site-packages/tensorrt_libs:${LD_LIBRARY_PATH}"

echo ""
echo "LD_LIBRARY_PATH set to:"
echo "  $LD_LIBRARY_PATH"

echo ""
echo "Verifying libnvinfer.so.10 is accessible..."
if ldconfig -p 2>/dev/null | grep -q libnvinfer || [ -f "/home/msamuels/ai-inference/lib/python3.12/site-packages/tensorrt_libs/libnvinfer.so.10" ]; then
    echo "  ✓ libnvinfer.so.10 found"
else
    echo "  ⚠ libnvinfer.so.10 not in ldconfig, but should work via LD_LIBRARY_PATH"
fi

echo ""
echo "Running benchmark..."
echo "========================================"

python3 benchmark_llama1b_trt.py