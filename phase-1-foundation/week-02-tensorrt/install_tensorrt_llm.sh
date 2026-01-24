#!/bin/bash
# TensorRT-LLM Installation Script
# Run this in your ai-inference virtual environment

echo "========================================"
echo "TENSORRT-LLM INSTALLATION"
echo "========================================"

# Activate environment
source ~/rtx3090-ai-training/ai-inference/bin/activate

# Check CUDA version
echo ""
echo "[1/4] Checking CUDA version..."
nvcc --version | grep "release"

# Install TensorRT-LLM
# Note: This is a large package (~2GB)
echo ""
echo "[2/4] Installing TensorRT-LLM..."
echo "  (This may take 5-10 minutes)"
pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com

# Verify installation
echo ""
echo "[3/4] Verifying installation..."
python3 -c "import tensorrt_llm; print(f'TensorRT-LLM version: {tensorrt_llm.__version__}')"

# Check GPU compatibility
echo ""
echo "[4/4] Checking GPU compatibility..."
python3 -c "
import torch
import tensorrt_llm

print(f'PyTorch CUDA: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} ({props.total_memory/1e9:.1f} GB)')
print(f'TensorRT-LLM ready: OK')
"

echo ""
echo "========================================"
echo "INSTALLATION COMPLETE"
echo "========================================"
echo ""
echo "Next: Run the Llama conversion script"
```

Run the installation first. Let me know if it succeeds or if there are dependency conflicts.

---

## What TensorRT-LLM Does Differently

Instead of trying to trace the PyTorch graph, TensorRT-LLM:

1. **Parses model architecture** (knows Llama structure)
2. **Rebuilds in TensorRT primitives** (native GPU ops)
3. **Implements KV cache** as first-class citizen
4. **Uses optimized kernels** (FP16, INT8, FP8 attention)
```
Traditional (broken):
  PyTorch Model → Trace Graph → ONNX → TensorRT
                     ↑
                  FAILS HERE

TensorRT-LLM (works):
  HuggingFace Checkpoint → TRT-LLM Builder → TensorRT Engine
                              ↑
                    Knows model architecture,
                    builds native TensorRT ops