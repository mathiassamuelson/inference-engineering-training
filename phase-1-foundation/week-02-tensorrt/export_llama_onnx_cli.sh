#!/bin/bash
# Filename: export_llama_onnx_cli.sh
# Location: phase-1-foundation/week-02-tensorrt/
# Purpose: Export Llama 3.2 3B to ONNX using Optimum CLI

echo "========================================"
echo "ONNX EXPORT VIA OPTIMUM CLI"
echo "========================================"

cd ~/rtx3090-ai-training/phase-1-foundation/week-02-tensorrt
source ~/rtx3090-ai-training/ai-inference/bin/activate

# Force CPU-only export
export CUDA_VISIBLE_DEVICES=""

OUTPUT_DIR="./results/llama_onnx_cli"
mkdir -p $OUTPUT_DIR

echo ""
echo "[1/2] Starting ONNX export via CLI..."
echo "  Model: meta-llama/Llama-3.2-3B-Instruct"
echo "  Output: $OUTPUT_DIR"
echo "  Device: CPU only (CUDA_VISIBLE_DEVICES='')"
echo ""
echo "  This may take 10-20 minutes..."
echo ""

# Use optimum-cli for export
optimum-cli export onnx \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --task text-generation-with-past \
    --framework pt \
    --opset 17 \
    --device cpu \
    --fp16 \
    $OUTPUT_DIR

export_status=$?

echo ""
echo "[2/2] Checking results..."

if [ $export_status -eq 0 ]; then
    echo "  ✓ Export completed successfully!"
    echo ""
    echo "  Files created:"
    ls -lh $OUTPUT_DIR
else
    echo "  ✗ Export failed with status: $export_status"
fi

# Restore CUDA
unset CUDA_VISIBLE_DEVICES

echo ""
echo "========================================"