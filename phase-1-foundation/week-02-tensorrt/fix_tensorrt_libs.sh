#!/bin/bash
# Filename: fix_tensorrt_libs.sh
# Location: phase-1-foundation/week-02-tensorrt/
# Purpose: Find and configure TensorRT library paths

echo "========================================"
echo "TENSORRT LIBRARY PATH CONFIGURATION"
echo "========================================"

. ~/ai-inference/bin/activate

echo ""
echo "[1/5] Finding TensorRT pip package location..."
TRT_PIP_PATH=$(python3 -c "import tensorrt_libs; print(tensorrt_libs.__path__[0])" 2>/dev/null)

if [ -z "$TRT_PIP_PATH" ]; then
    # Try alternative package name
    TRT_PIP_PATH=$(python3 -c "import tensorrt; import os; print(os.path.dirname(tensorrt.__file__))" 2>/dev/null)
fi

if [ -n "$TRT_PIP_PATH" ]; then
    echo "  Found: $TRT_PIP_PATH"
else
    echo "  Not found via pip package"
fi

echo ""
echo "[2/5] Searching for libnvinfer.so.10..."
echo "  Searching system paths..."

NVINFER_PATHS=$(find /usr -name "libnvinfer.so*" 2>/dev/null | head -5)
if [ -n "$NVINFER_PATHS" ]; then
    echo "  Found in /usr:"
    echo "$NVINFER_PATHS" | sed 's/^/    /'
fi

echo ""
echo "  Searching pip site-packages..."
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
PIP_NVINFER=$(find "$SITE_PACKAGES" -name "libnvinfer.so*" 2>/dev/null | head -5)
if [ -n "$PIP_NVINFER" ]; then
    echo "  Found in site-packages:"
    echo "$PIP_NVINFER" | sed 's/^/    /'
fi

echo ""
echo "  Searching home directory..."
HOME_NVINFER=$(find ~/ai-inference -name "libnvinfer.so*" 2>/dev/null | head -5)
if [ -n "$HOME_NVINFER" ]; then
    echo "  Found in ~/ai-inference:"
    echo "$HOME_NVINFER" | sed 's/^/    /'
fi

echo ""
echo "[3/5] Searching for all TensorRT-related libs..."
ALL_TRT_LIBS=$(find ~/ai-inference -name "*nvinfer*" -o -name "*tensorrt*" 2>/dev/null | grep "\.so" | head -10)
if [ -n "$ALL_TRT_LIBS" ]; then
    echo "  TensorRT libraries found:"
    echo "$ALL_TRT_LIBS" | sed 's/^/    /'

    # Extract unique directories
    LIB_DIRS=$(echo "$ALL_TRT_LIBS" | xargs -n1 dirname | sort -u)
    echo ""
    echo "  Library directories:"
    echo "$LIB_DIRS" | sed 's/^/    /'
fi

echo ""
echo "[4/5] Current LD_LIBRARY_PATH..."
echo "  $LD_LIBRARY_PATH"

echo ""
echo "[5/5] Checking ONNX Runtime TensorRT provider requirements..."
python3 << 'EOF'
import onnxruntime as ort

print("ONNX Runtime version:", ort.__version__)
print("Available providers:", ort.get_available_providers())

# Check if TensorRT provider can be loaded
if 'TensorrtExecutionProvider' in ort.get_available_providers():
    print("\n✓ TensorrtExecutionProvider is listed as available")
    print("  But may fail at runtime if libs not in LD_LIBRARY_PATH")
EOF

echo ""
echo "========================================"
echo "RECOMMENDED NEXT STEPS"
echo "========================================"