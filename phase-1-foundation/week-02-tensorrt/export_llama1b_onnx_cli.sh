#!/bin/bash
# Filename: export_llama1b_onnx_cli.sh
# Location: phase-1-foundation/week-02-tensorrt/
# Purpose: Export Llama 3.2 1B to ONNX (smaller model to verify pipeline)

echo "========================================"
echo "ONNX EXPORT: LLAMA 3.2 1B (PIPELINE TEST)"
echo "========================================"

cd ~/work/rtx3090-ai-training/phase-1-foundation/week-02-tensorrt
. ~/ai-inference/bin/activate

# Force CPU-only export
export CUDA_VISIBLE_DEVICES=""

OUTPUT_DIR="./results/llama1b_onnx_cli"
mkdir -p $OUTPUT_DIR

echo ""
echo "Testing with smaller model to verify ONNX pipeline..."
echo ""
echo "  Model: meta-llama/Llama-3.2-1B-Instruct"
echo "  Output: $OUTPUT_DIR"
echo "  Expected time: 5-10 minutes"
echo ""

# Monitor memory in background
echo "Starting memory monitor..."
(
    while true; do
        mem_used=$(free -g | awk '/^Mem:/{print $3}')
        echo "  [Memory: ${mem_used}GB used]"
        sleep 10
    done
) &
MONITOR_PID=$!

# Export 1B model
optimum-cli export onnx \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --task text-generation-with-past \
    --framework pt \
    --opset 17 \
    --device cpu \
    $OUTPUT_DIR

export_status=$?

# Stop monitor
kill $MONITOR_PID 2>/dev/null

echo ""
echo "========================================"
echo "RESULTS"
echo "========================================"

if [ $export_status -eq 0 ]; then
    echo "✓ Export completed successfully!"
    echo ""
    echo "Files created:"
    ls -lh $OUTPUT_DIR
    echo ""
    echo "Next: Run benchmark with this ONNX model"
else
    echo "✗ Export failed with status: $export_status"
    echo ""
    if [ $export_status -eq 137 ]; then
        echo "Status 137 = OOM killed"
        echo "Even 1B model needs too much RAM for ONNX export"
    fi
fi

# Restore CUDA
unset CUDA_VISIBLE_DEVICES

echo "========================================"