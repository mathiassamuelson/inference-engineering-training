#!/bin/bash
# Filename: diagnose_cuda_mismatch.sh
# Location: phase-1-foundation/week-02-tensorrt/
# Purpose: Diagnose CUDA version mismatch with TensorRT-LLM

echo "========================================"
echo "CUDA VERSION DIAGNOSTIC"
echo "========================================"

echo ""
echo "[1/4] Installed CUDA version..."
nvcc --version | grep "release"

echo ""
echo "[2/4] CUDA runtime libraries available..."
ls -la /usr/local/cuda/lib64/libcudart.so* 2>/dev/null
ls -la /usr/lib/x86_64-linux-gnu/libcudart.so* 2>/dev/null

echo ""
echo "[3/4] TensorRT-LLM package info..."
source ~/rtx3090-ai-training/ai-inference/bin/activate
pip show tensorrt-llm | grep -E "^(Name|Version|Requires)"

echo ""
echo "[4/4] What libcudart versions exist..."
find /usr -name "libcudart.so*" 2>/dev/null | head -20

echo ""
echo "========================================"
echo "DIAGNOSTIC COMPLETE"
echo "========================================"
echo ""
echo "The issue: TensorRT-LLM wants libcudart.so.13 (CUDA 13)"
echo "You have: CUDA 12.6 (libcudart.so.12)"
echo ""
echo "Options:"
echo "  1. Install TensorRT-LLM version for CUDA 12"
echo "  2. Use Docker container (NVIDIA's recommended approach)"
echo "  3. Use Optimum-NVIDIA instead (simpler integration)"
echo "  4. Skip TensorRT-LLM, document the complexity"