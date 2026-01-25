#!/bin/bash

echo "================================================"
echo "RTX 3090 AI Training - Environment Setup"
echo "================================================"

# Use ~/ai-inference as the standard environment location
VENV_PATH="$HOME/ai-inference"

#######################################################################
# SECTION 1: System Dependencies
#######################################################################
echo ""
echo "[1/6] Installing system dependencies..."

sudo apt-get update
sudo apt-get install -y \
    openmpi-bin \
    libopenmpi-dev \
    build-essential

#######################################################################
# SECTION 2: Python Environment
#######################################################################
echo ""
echo "[2/6] Setting up Python environment..."

python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "  Python version: $python_version"

if [ ! -d "$VENV_PATH" ]; then
    echo "  Creating virtual environment at $VENV_PATH..."
    python3 -m venv "$VENV_PATH"
else
    echo "  Virtual environment already exists at $VENV_PATH"
fi

. "$VENV_PATH/bin/activate"

# Upgrade pip
echo "  Upgrading pip..."
pip install --upgrade pip

#######################################################################
# SECTION 3: PyTorch with CUDA
#######################################################################
echo ""
echo "[3/6] Installing PyTorch with CUDA 12.1..."

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

#######################################################################
# SECTION 4: Core Dependencies
#######################################################################
echo ""
echo "[4/6] Installing core dependencies..."

pip install -r requirements.txt

# ONNX Runtime GPU (requires special index for CUDA 12)
echo "  Installing ONNX Runtime GPU..."
pip install optimum[onnxruntime-gpu]
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

#######################################################################
# SECTION 5: Environment Configuration
#######################################################################
echo ""
echo "[5/6] Configuring environment..."

# Add TensorRT libraries to LD_LIBRARY_PATH in .bashrc
TRT_LIB_PATH="$VENV_PATH/lib/python3.12/site-packages/tensorrt_libs"

if ! grep -q "tensorrt_libs" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# TensorRT libraries for ONNX Runtime" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\"$TRT_LIB_PATH:\${LD_LIBRARY_PATH}\"" >> ~/.bashrc
    echo "  ✓ Added TensorRT path to ~/.bashrc"
else
    echo "  ✓ TensorRT path already in ~/.bashrc"
fi

# Apply to current session
export LD_LIBRARY_PATH="$TRT_LIB_PATH:${LD_LIBRARY_PATH}"

#######################################################################
# SECTION 6: Directory Structure & Verification
#######################################################################
echo ""
echo "[6/6] Creating directory structure and verifying setup..."

# Create directory structure
mkdir -p docs/weekly-reports
mkdir -p phase-1-foundation/week-01-benchmarks/results
mkdir -p phase-1-foundation/week-01-benchmarks/profiles
mkdir -p phase-1-foundation/week-02-tensorrt/results
mkdir -p tools
mkdir -p notebooks/exploratory
mkdir -p assets/performance-charts

# Verify CUDA
echo ""
echo "  Verifying CUDA setup..."
python3 -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPU count: {torch.cuda.device_count()}')"

# Verify ONNX Runtime providers
echo ""
echo "  Verifying ONNX Runtime providers..."
python3 -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print(f'  Available: {providers}')
if 'TensorrtExecutionProvider' in providers:
    print('  ✓ TensorRT provider available')
if 'CUDAExecutionProvider' in providers:
    print('  ✓ CUDA provider available')
"

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "To activate the environment:"
echo "  source ~/ai-inference/bin/activate"
echo ""
echo "If TensorRT libraries aren't found, restart your shell or run:"
echo "  source ~/.bashrc"