#!/usr/bin/env python3
# Filename: check_onnx_providers.py
# Location: phase-1-foundation/week-02-tensorrt/
# Purpose: Check available ONNX Runtime execution providers

import warnings

warnings.filterwarnings("ignore")

print("=" * 50)
print("ONNX RUNTIME PROVIDER CHECK")
print("=" * 50)

import onnxruntime as ort

print(f"\nONNX Runtime version: {ort.__version__}")
print(f"\nAvailable execution providers:")

providers = ort.get_available_providers()
for p in providers:
    print(f"  ✓ {p}")

print("\n" + "-" * 50)
print("Provider analysis:")
print("-" * 50)

if "TensorrtExecutionProvider" in providers:
    print("  ✓ TensorRT available - can use TRT optimization")
else:
    print("  ✗ TensorRT NOT available")

if "CUDAExecutionProvider" in providers:
    print("  ✓ CUDA available - can use GPU acceleration")
else:
    print("  ✗ CUDA NOT available")

if "CPUExecutionProvider" in providers:
    print("  ✓ CPU available - fallback option")

print("\n" + "=" * 50)
