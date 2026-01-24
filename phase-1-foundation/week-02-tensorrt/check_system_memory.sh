#!/bin/bash
# Filename: check_system_memory.sh
# Location: phase-1-foundation/week-02-tensorrt/
# Purpose: Check system RAM and swap for ONNX export feasibility

echo "========================================"
echo "SYSTEM MEMORY CHECK"
echo "========================================"

echo ""
echo "[1/4] Total system memory..."
free -h

echo ""
echo "[2/4] Memory breakdown..."
cat /proc/meminfo | grep -E "^(MemTotal|MemFree|MemAvailable|SwapTotal|SwapFree)"

echo ""
echo "[3/4] Current memory usage by top processes..."
ps aux --sort=-%mem | head -10

echo ""
echo "[4/4] Available for ONNX export..."
available_gb=$(free -g | awk '/^Mem:/{print $7}')
echo "Available RAM: ${available_gb} GB"
echo ""
echo "ONNX export for Llama 3.2 3B typically needs: 25-35 GB RAM"
echo ""

if [ "$available_gb" -lt 25 ]; then
    echo "⚠️  WARNING: May not have enough RAM for ONNX export"
    echo ""
    echo "Options:"
    echo "  1. Close other applications"
    echo "  2. Add swap space (temporary)"
    echo "  3. Use smaller model for testing"
    echo "  4. Skip ONNX/TensorRT path (document findings)"
fi

echo "========================================"