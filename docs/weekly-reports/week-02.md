# Week 2: TensorRT Optimization Experiments

**Duration:** January 20-26, 2026  
**Hardware:** 4x RTX 3090 (96GB total VRAM), Ubuntu 24.04 LTS, CUDA 12.6  
**Focus:** TensorRT installation, conversion pipeline, and optimization fundamentals

---

## Executive Summary

Week 2 established the TensorRT optimization pipeline and revealed critical insights about when GPU optimizations provide meaningful speedups:

1. **TensorRT speedup scales with model size** - SimpleNet (1.06M params) showed only 1.17x improvement because overhead dominated the 30-microsecond inference time. For Llama 3.2 3B with 12ms inference, overhead becomes negligible (4% vs 33%), enabling realistic 1.4-1.5x speedups.

2. **Kernel fusion reduces memory traffic** - Combining operations (e.g., Linear+ReLU) into single GPU kernels eliminates intermediate VRAM writes. For memory-bound models like Llama, this can reduce total bandwidth consumption by 20-30%.

3. **Memory bandwidth remains the bottleneck** - Week 1 showed FP16 achieving 54% of peak bandwidth (504/936 GB/s). TensorRT optimizations can improve this to ~70% (650 GB/s) through better memory layouts and fused attention kernels, but fundamental hardware limits still apply.

4. **Production frameworks solve different problems** - TensorRT optimizes single-request performance (1.4-1.5x), while vLLM addresses batch scaling issues discovered in Week 1. The batch scaling plateau (5,000 tok/s regardless of batch size) requires framework-level solutions, not just kernel optimizations.

**Key takeaway:** TensorRT is a necessary but insufficient optimization for production inference. It provides modest single-request improvements but doesn't solve the batch scaling crisis from Week 1. Phase 2 (vLLM) will address the real throughput bottleneck.

---

## Objectives

- ✅ Install TensorRT 10.x compatible with CUDA 12.6
- ✅ Verify all 4 RTX 3090 GPUs are operational
- ✅ Establish PyTorch → ONNX → TensorRT conversion pipeline
- ✅ Understand kernel fusion, memory layout, and attention optimizations
- ✅ Set realistic expectations for Llama 3.2 3B conversion (Experiment 2)

---

## Experiment 1: TensorRT Setup & Conversion Pipeline Verification

### Setup & Configuration

**Software Stack:**
- TensorRT 10.x (CUDA 12 compatible, pip-based installation)
- CUDA-X libraries: cuDNN, cuBLAS, NCCL
- ONNX ecosystem: onnx>=1.15.0, onnxscript>=0.1.0, onnxruntime>=1.16.0

**Multi-GPU Verification Results:**
- ✅ All 4 RTX 3090 GPUs detected (25.3 GB each = 101.2 GB total)
- ✅ Compute Capability: 8.6 (Ampere architecture)
- ✅ Multi-Processors: 82 per GPU
- ✅ Memory allocation and computation tests passed
- ✅ TensorRT can access all GPUs via CUDA runtime

### Simple Model Conversion Test

**Test Model: SimpleNet (1.06M parameters)**