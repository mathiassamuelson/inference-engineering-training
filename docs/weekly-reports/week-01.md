# Week 1: Foundation - Baseline Benchmarks & Scaling Analysis

**Date:** January 13-19, 2025  
**Hardware:** 2x RTX 3090 (24GB each), Ubuntu 24.04, CUDA 12.6

## Objectives
- Establish baseline inference performance
- Understand FP32 vs FP16 speedup
- Explore batch size scaling characteristics
- Determine optimal throughput vs latency tradeoffs

## Experiments Conducted

### 1. Precision Comparison (Llama 3.2 3B)

| Configuration | Throughput | Memory | Speedup |
|--------------|------------|--------|---------|
| Single GPU FP32 | 53.95 tok/s | 12.86 GB | 1.00x |
| Single GPU FP16 | 84.08 tok/s | 6.43 GB | 1.56x |
| Dual GPU (auto) | 83.95 tok/s | 6.43 GB | 1.00x |

**Key Finding:** FP16 provides only 1.56x speedup (not 2-3x) because inference is **memory bandwidth limited**, not compute limited.

### 2. Batch Size Scaling

Tested batch sizes from 1 to 1024, generating 50 tokens per sequence.

**Results:**
- **Peak throughput:** 4,952 tokens/sec at batch=1024
- **GPU saturation:** ~5000 tok/s represents compute ceiling
- **Memory usage:** Constant 6.43 GB across all batch sizes
- **Optimal batch size:** 32-64 for throughput/latency balance

**Scaling Efficiency:**
- Batch 1-32: 76-100% efficient (excellent)
- Batch 64: 62% efficient (good)
- Batch 128+: <40% efficient (diminishing returns)

### 3. Latency vs Throughput Tradeoff

| Batch Size | Total Throughput | Per-Request Latency | Use Case |
|------------|-----------------|---------------------|----------|
| 1-4 | 85-312 tok/s | 1.2-1.3s | Real-time chat |
| 8-16 | 608-1137 tok/s | 1.3-1.4s | API services |
| 32-64 | 2048-3374 tok/s | 1.6-1.9s | Batch processing |
| 128+ | 4287+ tok/s | 3.0+ seconds | Maximum throughput |

## Critical Insights

### 1. Memory Bandwidth is the Bottleneck
- Autoregressive decode loads entire model for each token
- FP16 reduces memory footprint but bandwidth still constrains
- RTX 3090: Achieving 54% of peak 936 GB/s bandwidth

### 2. Small Models Don't Benefit from Multi-GPU
- Device_map="auto" kept 3B model on single GPU (smart!)
- Cross-GPU communication overhead would hurt performance
- Multi-GPU only beneficial for models >20B

### 3. Massive Headroom Available
- Only using 6.43 GB of 24 GB available
- KV cache extremely small for 50-token generation
- Could run much longer contexts or larger models

### 4. GPU Compute Saturates at ~5000 tok/s
- Plateaus at batch=512-1024
- Represents maximum throughput for this model/hardware combo
- Cannot improve further without model optimization

## Cost Economics

**Cost per million tokens @ batch=64:**
- Compute: $0.0041
- Power: $0.0001
- **Total: $0.0042 per million tokens**

**Comparison:**
- OpenAI GPT-4o mini: $0.150/M tokens → **36x more expensive**
- Claude Haiku: $0.250/M tokens → **60x more expensive**

**Break-even analysis:**
- RTX 3090 purchase: ~$1,200 (used market)
- Break-even vs GPT-4o mini: 8.2B tokens
- At 10M tokens/day: ROI in 27 months
- At 100M tokens/day: ROI in 3 months

### Memory Scaling Discovery

Peak memory usage follows the equation:
```
Peak Memory ≈ 6.0 GB (model) + 0.3 GB (fixed) + 13.2 MB × batch_size (KV cache)
```

**Critical insight:** Fixed overhead (CUDA buffers, activation temps) gets **amortized** across batch size, making larger batches more memory-efficient per sample:
- Batch 1: 457 MB per sample
- Batch 1024: 13.5 MB per sample (34x more efficient!)

**Memory ceiling:** RTX 3090's 24 GB allows batch ~1400 before OOM

**Flash Attention detected:** Actual KV cache (13.5 MB/sample) is ~25% less than theoretical (17.2 MB/sample), indicating PyTorch is using optimized attention implementation.

### Implications for Production

1. **Batching is mandatory:** Running batch=1 wastes 34x memory per token
2. **Sweet spot:** Batch 64-128 balances throughput, latency, and memory
3. **Scaling strategy:** Add GPUs for model size, use batching for throughput
4. **Cost optimization:** Fixed overhead means marginal cost per token drops 34x with batching

## Product Implications

### For AI Engineers:
1. **Batch intelligently:** Use batch=32-64 for best throughput/latency
2. **Don't over-parallelize:** Small models run faster on single GPU
3. **Memory headroom:** Can serve multiple models or longer contexts

### For AI Product Managers:
1. **Pricing strategy:** Can't charge linearly with GPU count for small models
2. **SLA design:** Batch size directly impacts per-request latency
3. **Cost optimization:** Self-hosting becomes economical at >10M tokens/day

### For Infrastructure Planning:
1. **Scaling pattern:** Add GPUs for model size, not throughput (until batch saturation)
2. **Memory vs compute:** Most inference is bandwidth-bound
3. **Optimization priority:** Focus on reducing memory bandwidth needs

## Next Steps (Week 2)
- [ ] Implement TensorRT optimization
- [ ] Compare PyTorch vs TensorRT performance
- [ ] Profile with NVIDIA NSight
- [ ] Prepare for 4-GPU upgrade in Week 3

## Files Generated
- `baseline_benchmark.py` - FP16/FP32 comparison
- `batch_size_scaling.py` - Throughput scaling analysis
- `cost_analysis.py` - Economic modeling
- `results/week1_baseline.csv`
- `results/batch_size_scaling.csv`
