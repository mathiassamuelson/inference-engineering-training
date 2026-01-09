#!/bin/bash

echo "================================================"
echo "RTX 3090 AI Training - Environment Setup"
echo "================================================"

# Check for Python 3.10+
python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv ai-inference
source ai-inference/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 12.1..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Verify CUDA
echo ""
echo "Verifying CUDA setup..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p docs/weekly-reports
mkdir -p phase-1-foundation/week-01-benchmarks/results
mkdir -p phase-1-foundation/week-01-benchmarks/profiles
mkdir -p tools
mkdir -p notebooks/exploratory
mkdir -p assets/performance-charts

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo "Activate environment with: source ai-inference/bin/activate"
