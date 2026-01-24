#!/bin/bash
# Filename: install_optimum_nvidia.sh
# Location: phase-1-foundation/week-02-tensorrt/
# Purpose: Install Optimum-NVIDIA for TensorRT optimization via HuggingFace

echo "========================================"
echo "OPTIMUM-NVIDIA INSTALLATION"
echo "========================================"

# Activate environment
cd ~/rtx3090-ai-training
source ai-inference/bin/activate

# First, let's check what we have
echo ""
echo "[1/5] Current environment..."
echo "Python: $(python3 --version)"
echo "CUDA: $(nvcc --version | grep release | awk '{print $6}')"

# Install optimum with NVIDIA backend
echo ""
echo "[2/5] Installing Optimum-NVIDIA..."
pip install optimum[onnxruntime-gpu]

# Also need onnxruntime-gpu for CUDA 12
echo ""
echo "[3/5] Installing ONNX Runtime GPU (CUDA 12)..."
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# Verify installation
echo ""
echo "[4/5] Verifying installation..."
python3 << 'EOF'
import warnings
warnings.filterwarnings('ignore')

print("Checking imports...")

try:
    import optimum
    print(f"  ✓ Optimum version: {optimum.__version__}")
except ImportError as e:
    print(f"  ✗ Optimum: {e}")

try:
    import onnxruntime as ort
    print(f"  ✓ ONNX Runtime version: {ort.__version__}")
    providers = ort.get_available_providers()
    print(f"  ✓ Available providers: {providers}")
    if 'CUDAExecutionProvider' in providers:
        print("  ✓ CUDA execution available!")
    if 'TensorrtExecutionProvider' in providers:
        print("  ✓ TensorRT execution available!")
except ImportError as e:
    print(f"  ✗ ONNX Runtime: {e}")

try:
    import torch
    print(f"  ✓ PyTorch CUDA: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"  ✗ PyTorch: {e}")

print("\nDone!")
EOF

echo ""
echo "[5/5] Quick GPU test..."
python3 -c "
import torch
print(f'GPUs available: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo ""
echo "========================================"
echo "INSTALLATION COMPLETE"
echo "========================================"
echo ""
echo "Next: Run benchmark_optimum_trt.py"