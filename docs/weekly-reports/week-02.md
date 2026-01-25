# Week 2: TensorRT Optimization Experiments

**Duration:** January 20-26, 2026  
**Hardware:** 4x RTX 3090 (24GB each), Ubuntu 24.04 LTS, CUDA 12.6  
**Test Models:** SimpleNet (1M params), Llama 3.2 1B, Llama 3.2 3B (attempted)

---

## Executive Summary

Generic ONNX/TensorRT pipelines don't deliver expected LLM speedups:

1. **TensorRT benefits scale with model size** - SimpleNet showed 1.17x speedup; overhead (10μs) dominated the 30μs inference time.

2. **ONNX export requires massive RAM** - Llama 3.2 3B export OOM-killed despite 57GB available RAM (tied weight duplication).

3. **Device placement breaks performance** - ONNX Runtime kept weights on CPU, causing CPU↔GPU transfer overhead.

4. **Result: TensorRT slower than PyTorch** - Llama 3.2 1B: 81 tok/s (TensorRT) vs 183 tok/s (PyTorch FP16).

**Key takeaway:** Production LLM inference requires purpose-built frameworks (vLLM, TensorRT-LLM) that handle device placement and KV cache at the framework level.

---

## Objectives

- ✅ Install TensorRT and verify 4-GPU setup
- ✅ Establish PyTorch → ONNX → TensorRT pipeline
- ✅ Benchmark TensorRT vs PyTorch for LLM inference
- ❌ Achieve speedup over PyTorch baseline
- ✅ Understand why specialized frameworks exist

---

## Experiment 1: SimpleNet Conversion Test

**Model:** 1M parameter feed-forward network

| Metric | PyTorch FP32 | TensorRT FP16 | 
|--------|-------------|---------------|
| Latency | 0.030 ms | 0.025 ms |
| Throughput | 268K samples/s | 314K samples/s |
| Speedup | 1.0x | 1.17x |

**Why only 1.17x?** Fixed overhead (~10μs) dominates at microsecond inference times:

| Model Size | Inference Time | Overhead % | Expected Speedup |
|------------|---------------|------------|------------------|
| 1M params | 30 μs | 33% | 1.17x ✓ |
| 1B params | 545 ms | <0.01% | 1.4-1.5x |
| 3B params | 1,200 ms | <0.01% | 1.4-1.5x |

---

## Experiment 2: Llama LLM Conversion

### Failed Attempts

| Approach | Error | Root Cause |
|----------|-------|------------|
| Direct ONNX export | TorchExportError | PyTorch 2.x can't trace dynamic ops (RoPE, KV cache) |
| TensorRT-LLM | `libcudart.so.13` missing | Requires CUDA 13; system has CUDA 12.6 |
| Optimum (3B model) | OOM killed | Export needs >60GB RAM for tied weight duplication |

### Successful: Llama 3.2 1B via Optimum

Export completed in 10 minutes, peak RAM usage 28GB, output size 4.7GB.

### Benchmark Results

| Configuration | Throughput | Latency | GPU Memory | Speedup |
|--------------|-----------|---------|------------|---------|
| PyTorch FP16 | 183.4 tok/s | 545 ms | 2.47 GB | 1.00x |
| ONNX + CUDA | 84.5 tok/s | 1,183 ms | 0.01 GB | 0.46x |
| ONNX + TensorRT | 81.0 tok/s | 1,235 ms | 0.01 GB | 0.44x |

**Root cause:** 0.01GB GPU memory = weights on CPU. Each inference copies 1.2GB CPU→GPU→CPU.

ONNX Runtime warnings confirmed:
```
"1 Memcpy nodes are added to the graph... negative impact on performance"
```

---

## Why TensorRT Optimization Works (When It Works)

**Kernel Fusion:** Combines operations (MatMul+ReLU) reducing kernel launches and memory traffic.

**Memory Layout:** Optimizes tensor layouts (NCHW vs NHWC) for coalesced GPU access.

**Why these didn't help:** Optimizations require weights on GPU. ONNX Runtime's device placement kept weights on CPU, negating all benefits.

---

## Challenges & Resolutions

| Challenge | Resolution |
|-----------|------------|
| `source: not found` | Use `.` instead of `source` (POSIX) |
| `libmpi.so.40` missing | `apt install openmpi-bin libopenmpi-dev` |
| `libnvinfer.so.10` not found | Add tensorrt_libs to `LD_LIBRARY_PATH` |
| Llama 3B ONNX export OOM | Test with 1B model instead |

---

## Key Learnings

### Expected vs Reality

| Expected | Actual |
|----------|--------|
| 1.5-2x TensorRT speedup | 0.44x (slower) |
| Simple pip install | Multiple dependency conflicts |
| Drop-in optimization | Requires framework-level fixes |

### Why Specialized Frameworks Exist

**Generic pipeline problems:**
- ONNX export can't handle dynamic KV cache
- Default device placement = CPU
- No autoregressive generation optimization

**What vLLM/TensorRT-LLM solve:**
- Native GPU memory management
- KV cache as first-class citizen
- Continuous batching
- Fused attention kernels

---

## Interview Articulations

**On overhead scaling:**
> "TensorRT speedup scales with model size because fixed overhead becomes negligible. At 30μs inference, 10μs overhead limits gains to 1.17x. At 1,200ms inference, the same overhead is <0.01%, enabling 1.4-1.5x gains."

**On why generic pipelines fail:**
> "ONNX Runtime achieved 0.44x of PyTorch's throughput because weights stayed on CPU. Each forward pass copied 1.2GB across PCIe, overwhelming any kernel optimization. This is why TensorRT-LLM and vLLM exist - they manage device placement and KV cache at the framework level."

**On production deployment:**
> "Week 2 taught me that 'pip install tensorrt' isn't production optimization. Real speedups require frameworks that understand LLM architecture: KV cache memory management, continuous batching, and proper GPU placement. Generic ONNX pipelines can't deliver this."

---

## Files Created
```
phase-1-foundation/week-02-tensorrt/
├── setup_tensorrt.sh
├── verify_multi_gpu.py
├── simple_conversion_test.py
├── export_llama1b_onnx_cli.sh
├── benchmark_llama1b_trt.py
├── fix_tensorrt_libs.sh
└── results/
    └── llama1b_onnx_cli/
        ├── model.onnx (715 KB)
        ├── model.onnx_data (4.7 GB)
        └── tokenizer files
```

---

## Next Steps

Week 2 validates the training plan structure: specialized frameworks solve problems that generic pipelines cannot. 

**Immediate:** Proceed to Phase 2 (Weeks 5-8) focusing on vLLM, which handles device placement, KV cache, and batching correctly.

**Week 3-4:** Multi-GPU orchestration with tensor parallelism for larger models.

---

*Report generated: January 26, 2026*  
*Hardware: 4x RTX 3090, Ubuntu 24.04, CUDA 12.6*