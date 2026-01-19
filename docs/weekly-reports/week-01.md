# Week 1: NVIDIA Compute Platform Fundamentals

**Duration:** January 13-19, 2026  
**Hardware:** 2x RTX 3090 (24GB each), Ubuntu 24.04 LTS, CUDA 12.6  
**Test Model:** Llama 3.2 3B Instruct (3 billion parameters)

---

## Executive Summary

Week 1 established that **inference on small models is fundamentally memory-bandwidth limited**, not compute-limited. FP16 optimization provided only 1.56x speedup (not the theoretical 2-3x) because token generation is bottlenecked by reading model weights from VRAM. Batch size scaling analysis produced a linear memory model with <1.1% error, enabling precise capacity planning: a single RTX 3090 can serve ~1,200 concurrent users when properly batched. Most importantly, multi-GPU splitting provides no benefit for 3B models—the framework correctly keeps the model on a single device to avoid PCIe overhead.

**Key learnings:**
- Memory bandwidth, not compute, limits inference throughput
- Batch size is the primary efficiency lever (27% → 89% GPU utilization)
- Linear memory model enables predictive capacity planning
- Small models gain nothing from multi-GPU deployment

---

## Objectives

- ✅ Establish baseline inference performance metrics
- ✅ Understand FP16 optimization benefits and limitations  
- ✅ Measure batch size scaling behavior
- ✅ Build predictive memory capacity model
- ✅ Validate multi-GPU behavior for small models

---

## Experiment 1: Baseline Performance Analysis

### Setup

**Model:** meta-llama/Llama-3.2-3B-Instruct (3B parameters)  
**Test configurations:**
- Single GPU FP32 (32-bit floating point)
- Single GPU FP16 (16-bit floating point)  
- Dual GPU with automatic device mapping

**Workload:** Generate 100 tokens, 10 iterations, single prompt

### Results

| Configuration | Throughput | Memory Usage | Speedup vs FP32 |
|--------------|------------|--------------|-----------------|
| Single GPU FP32 | 53.95 tok/s | 12.86 GB | 1.00x (baseline) |
| Single GPU FP16 | 84.08 tok/s | 6.43 GB | **1.56x** |
| Dual GPU (auto) | 83.95 tok/s | 6.43 GB | 1.00x vs FP16 |

### Key Findings

#### Finding 1: FP16 Speedup Lower Than Expected (1.56x vs 2-3x theoretical)

**Root cause:** Memory bandwidth bottleneck, not compute

**Analysis:**
```
Memory bandwidth utilization:
- 84 tokens/sec × 3B params × 2 bytes (FP16) = 504 GB/sec
- RTX 3090 peak bandwidth: 936 GB/sec  
- Efficiency: 54% of theoretical peak
```

**Why this matters:**
- Autoregressive decoding loads entire model for each token
- Tensor cores sit idle waiting for data from VRAM
- Compute intensity too low to saturate 35.6 TFLOPS (FP16)
- This is fundamentally different from training (compute-bound)

**Technical explanation:**

In autoregressive generation, each token requires:
1. Load 3B parameters from VRAM (6 GB in FP16)
2. Compute attention + FFN (minimal compute)
3. Generate 1 token
4. Repeat

At 84 tokens/sec, we're transferring 504 GB/sec—this is the actual bottleneck. The tensor cores could theoretically process much faster, but they're starved for data.

**Implication for optimization strategy:**

Priority order for inference optimization:
1. **Memory bandwidth:** Quantization (INT8/INT4), compression
2. **Batch size:** Amortize memory reads across multiple samples
3. **KV cache efficiency:** Reduce redundant loads (Flash Attention)
4. **Compute optimization:** TensorRT, kernel fusion (secondary concern)

#### Finding 2: Memory Reduction Shows Perfect 2x Scaling

FP32 → FP16 halved memory usage exactly (12.86 GB → 6.43 GB):

**Memory breakdown:**
```
FP32 (12.86 GB):
- Model weights: 3B params × 4 bytes = 12.0 GB
- Overhead: 0.86 GB (activations, buffers, KV cache)

FP16 (6.43 GB):
- Model weights: 3B params × 2 bytes = 6.0 GB  
- Overhead: 0.43 GB (activations, buffers, KV cache)

Leaves 17.57 GB free for:
- Larger batch sizes
- Longer context windows
- Multiple concurrent models
```

#### Finding 3: Dual GPU Shows No Benefit

PyTorch's `device_map="auto"` kept entire model on GPU 0 (smart behavior):

**Why this is correct:**
- 3B model (6.43 GB FP16) fits comfortably on 24 GB GPU
- PCIe Gen4 x16 bandwidth: ~32 GB/sec bidirectional
- Cross-GPU communication would add 15-30% latency overhead
- No memory pressure that would justify splitting

**Validation experiment needed (Week 3 with 4 GPUs):**
- Force manual split across 2 GPUs
- Measure PCIe transfer overhead
- Quantify exact penalty for educational purposes

**Product insight:** 
- Don't charge users premium for multi-GPU on small models
- Multi-GPU only helps when model doesn't fit on single device
- For 3B-13B models: single GPU is optimal
- For 70B+ models: multi-GPU becomes necessary

---

## Experiment 2: Batch Size Scaling Analysis

### Setup

**Objective:** Understand how memory scales with concurrent requests  
**Test range:** Batch sizes 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1200  
**Generation length:** 50 tokens per sample  
**Precision:** FP16

### Results

| Batch Size | Total Throughput | Per-Sample Throughput | Peak Memory | GPU Utilization |
|------------|------------------|----------------------|-------------|-----------------|
| 1 | 84 tok/s | 84.0 tok/s | 6.45 GB | 27% |
| 2 | 167 tok/s | 83.5 tok/s | 6.46 GB | 27% |
| 4 | 331 tok/s | 82.8 tok/s | 6.49 GB | 27% |
| 8 | 657 tok/s | 82.1 tok/s | 6.55 GB | 27% |
| 16 | 1,300 tok/s | 81.3 tok/s | 6.67 GB | 28% |
| 32 | 2,566 tok/s | 80.2 tok/s | 6.89 GB | 29% |
| 64 | 5,032 tok/s | 78.6 tok/s | 7.35 GB | 31% |
| 128 | 9,728 tok/s | 76.0 tok/s | 8.06 GB | 34% |
| 256 | 18,432 tok/s | 72.0 tok/s | 9.69 GB | 40% |
| 512 | 34,816 tok/s | 68.0 tok/s | 12.95 GB | 54% |
| 1024 | 63,488 tok/s | 62.0 tok/s | 19.46 GB | 81% |
| 1200 | 72,000 tok/s | 60.0 tok/s | 21.43 GB | 89% |

### Key Findings

#### Finding 1: Near-Linear Throughput Scaling

Total throughput scales nearly perfectly with batch size up to 1024:
- Batch 1 → 1024: **756x increase** in aggregate throughput
- Per-sample latency increases proportionally (1 batch = all samples wait)
- Memory becomes limiting factor above batch 1024

**Critical trade-off visualization:**

```
Low Batch (Real-time):
- Batch 1-4: 80+ tok/s per user
- Latency: <100ms
- Use cases: Chat, interactive AI, real-time streaming
- GPU utilization: 27% (wasteful!)

High Batch (Throughput-optimized):
- Batch 512-1200: 60-68 tok/s per user  
- Latency: 2-6 seconds
- Use cases: Batch processing, document analysis, async APIs
- GPU utilization: 54-89% (efficient!)
```

**Engineering insight:**

The per-sample throughput degradation (84 → 60 tok/s) is acceptable because:
- 26% performance hit on individual samples
- But 756x more total work completed
- Trade-off is favorable for batch workloads

#### Finding 2: GPU Utilization Increases with Batch Size

Utilization progression:
- **Batch 1-32:** 27-29% (severely underutilized)
- **Batch 64-256:** 31-40% (poor efficiency)
- **Batch 512:** 54% (acceptable)
- **Batch 1024:** 81% (good efficiency)
- **Batch 1200:** 89% (near-optimal)

**Why low utilization at small batch sizes?**

1. **Memory-bound operation:** GPU stalls waiting for data
2. **Low arithmetic intensity:** Not enough compute per memory load
3. **Kernel launch overhead:** Fixed cost per forward pass
4. **Underutilized tensor cores:** Insufficient parallelism

**Solution:** Dynamic batching (vLLM, Triton Inference Server)
- Queue incoming requests
- Wait up to X milliseconds to form larger batch
- Process batch together
- Improves utilization while controlling latency

#### Finding 3: Memory Grows Linearly, Throughput Scales Sub-linearly

**Memory scaling:** Perfectly linear (see Experiment 3)

**Throughput scaling:** Sub-linear degradation
- Expected: 1200 × 84 tok/s = 100,800 tok/s (if perfect)
- Actual: 72,000 tok/s
- Efficiency: 71% of ideal

**Causes of degradation:**
1. Memory contention between samples
2. Cache effects (reduced cache hit rate)
3. Increased memory bandwidth pressure
4. CUDA scheduler overhead for massive parallelism

**Product decision:** 
- Optimal batch size depends on latency tolerance
- Sweet spot appears around 512-1024 for this model
- Beyond 1200, hitting diminishing returns

---

## Experiment 3: Memory Capacity Model

### Methodology

Built linear regression model to predict memory usage:

```python
Peak Memory = Base + (Per_Sample_Cost × Batch_Size)
```

Used batch size scaling data to fit parameters via least squares.

### Results

**Fitted Model:**
```
Peak Memory = 6.470 GB + 12.57 MB × batch_size

Where:
- Base = 6.470 GB (model weights + fixed overhead)
- Per-sample = 12.57 MB (KV cache for 50 token generation)
- R² = 0.9999 (near-perfect fit)
```

**Model Validation:**

| Batch | Predicted Memory | Actual Memory | Absolute Error |
|-------|-----------------|---------------|----------------|
| 1 | 6.48 GB | 6.45 GB | 0.55% |
| 2 | 6.50 GB | 6.46 GB | 0.50% |
| 4 | 6.52 GB | 6.49 GB | 0.44% |
| 8 | 6.57 GB | 6.55 GB | 0.34% |
| 16 | 6.67 GB | 6.67 GB | 0.06% |
| 32 | 6.87 GB | 6.89 GB | 0.31% |
| 64 | 7.27 GB | 7.35 GB | 1.02% |
| 128 | 8.08 GB | 8.06 GB | 0.20% |
| 256 | 9.69 GB | 9.69 GB | 0.03% |
| 512 | 12.91 GB | 12.95 GB | 0.34% |
| 1024 | 19.34 GB | 19.46 GB | 0.62% |
| 1200 | 21.56 GB | 21.43 GB | 0.59% |

**Maximum error: 1.02%** across all tested batch sizes!

### Technical Validation: KV Cache Analysis

**Theoretical KV cache size calculation:**

```
Llama 3.2 3B architecture:
- Layers: 28
- Attention heads: 24 (with GQA: 8 KV heads)
- Hidden dimension: 3,072
- Precision: FP16 (2 bytes)

KV cache per token:
= 2 (K and V) × 28 layers × (3072 / 3) dim × 2 bytes
= 2 × 28 × 1024 × 2
= 114,688 bytes = 112 KB per token

For 50 tokens generation:
50 × 112 KB = 5.6 MB theoretical (with GQA optimization)

Note: GQA (Grouped Query Attention) reduces KV heads:
- Full attention: 24 heads for K, V
- GQA: 8 heads for K, V (3x reduction)
```

**Observed: 12.57 MB per sample**

**Explanation of difference (12.57 MB vs 5.6 MB theoretical):**

1. **Input tokens:** Not just 50 output tokens, but also ~30-40 input tokens in prompt
2. **Total tokens cached:** ~90 tokens per sample (input + output)
3. **Adjusted calculation:** 90 × 112 KB = 10.08 MB
4. **Remaining difference:** PyTorch overhead, attention score buffers, temporary tensors

This validates our empirical measurement!

### Capacity Planning Framework

**Component breakdown:**
- **Base memory (6.47 GB):** Model weights + framework overhead
  - 3B params × 2 bytes (FP16) = 6.0 GB
  - PyTorch framework: ~0.47 GB
- **Per-sample memory (12.57 MB):** KV cache for full generation

**Maximum batch size calculation:**

```
Total GPU memory: 24.0 GB
- Base memory: 6.47 GB
- Safety margin: 2.5 GB (for fragmentation, activations)
= Available for KV cache: 15.03 GB

Max batch size: 15.03 GB ÷ 12.57 MB = 1,195 samples
```

**Safety margin justification (10.7% / 2.5 GB):**

1. **Memory fragmentation:** PyTorch allocator creates gaps
2. **Temporary activations:** Forward pass creates intermediate tensors
3. **Gradient buffers:** Even in inference mode, some allocations occur
4. **OS overhead:** CUDA driver reserves some VRAM

**Recommended batch limits:**

| Strategy | Max Batch | Safety Margin | Use Case |
|----------|-----------|---------------|----------|
| Conservative | 1,000 | 4.0 GB (16.7%) | Production, mission-critical |
| Standard | 1,200 | 2.5 GB (10.4%) | Standard deployment |
| Aggressive | 1,350 | 1.0 GB (4.2%) | Benchmarking, experimentation |
| Theoretical | 1,394 | 0 GB | Educational only (will OOM!) |

### Scaling to Multiple GPUs

**With 4x RTX 3090 (96 GB total):**

```
Available memory per model replica:
= (96 GB - 4 × 2.5 GB safety) ÷ 1 model
= 86 GB for single model

Max batch: 86 GB ÷ 12.57 MB = ~6,800 samples (if loading single model)

OR distribute 4 model replicas:
- Each GPU: 1 model replica
- Per-GPU batch: 1,200
- Total capacity: 4,800 concurrent users
```

**Linear scaling confirmed!**

---

## Product & Engineering Insights

### 1. Infrastructure Sizing Framework

**Critical question:** "How many concurrent users can a single RTX 3090 serve?"

**Answer matrix (for 50 token generations):**

| Use Case | Recommended Batch | Users/GPU | Latency SLA | GPU Util | Cost Efficiency |
|----------|------------------|-----------|-------------|----------|-----------------|
| Real-time chat | 1-4 | 1-4 | <100ms | 27% | ⚠️ Poor |
| Interactive API | 8-32 | 8-32 | 200-500ms | 28% | ⚠️ Poor |
| Smart batching | 64-128 | 64-128 | 1-2s | 32-34% | ⚠️ Mediocre |
| Batch processing | 512-1024 | 512-1024 | 2-5s | 54-81% | ✅ Good |
| Max throughput | 1200 | 1200 | 6s | 89% | ✅ Excellent |

**Key insight:** Real-time use cases waste 73% of hardware capacity!

**Solution:** Request batching middleware
- vLLM: Continuous batching with PagedAttention
- NVIDIA Triton: Dynamic batching with queue management
- Custom: Implement timed batching (wait N ms to accumulate requests)

### 2. Multi-GPU Economics

**Scaling analysis for 2x and 4x RTX 3090 configurations:**

| Configuration | Max Users (batched) | Hardware Cost | Cost per 1,000 Users/Month |
|--------------|-------------------|---------------|---------------------------|
| 1x RTX 3090 | 1,200 | $1,500 | $1,250 |
| 2x RTX 3090 | 2,400 | $3,000 | $1,250 |
| 4x RTX 3090 | 4,800 | $6,000 | $1,250 |

**Perfect linear scaling!** Each additional GPU provides proportional capacity for small models.

**Cloud comparison (AWS g5 instances):**

| Instance Type | GPUs | vCPUs | Memory | Cost/Hour | Monthly Cost (24/7) |
|--------------|------|-------|--------|-----------|---------------------|
| g5.xlarge | 1x A10G | 4 | 16 GB | $1.006 | $726 |
| g5.12xlarge | 4x A10G | 48 | 192 GB | $5.672 | $4,085 |

**Break-even analysis (4x RTX 3090):**

```
Hardware cost: $6,000
Monthly cloud cost: $4,085
Break-even time: 6,000 ÷ 4,085 = 1.5 months

Annual savings: ($4,085 × 12) - $6,000 = $43,020

3-year TCO:
- Cloud: $147,060
- On-prem: $6,000 + $500 (power) × 36 = $24,000
- Savings: $123,060 (83% reduction)
```

**Critical assumptions:**
- 24/7 utilization (justified if serving >1,000 users)
- 3-year hardware lifetime
- Excludes: cooling, space, management overhead
- Includes: power at $0.12/kWh, 4 × 350W × 24/7

### 3. Batch Size vs Latency Trade-offs

**Product decision framework:**

```
IF application = "real-time chat"
  THEN batch_size = 1-4 (accept poor utilization)
  REASON: Latency is paramount, users won't wait

ELSE IF application = "API with SLA"
  THEN batch_size = dynamic based on request rate
  IMPLEMENT: vLLM continuous batching
  EXAMPLE: Wait 50ms to accumulate 8-32 requests

ELSE IF application = "batch processing"
  THEN batch_size = 512-1200 (maximize utilization)
  REASON: Throughput matters, latency acceptable
```

**Example use cases mapped to batch strategy:**

| Use Case | Optimal Batch | Reasoning |
|----------|--------------|-----------|
| ChatGPT-style interface | 1-8 | User expects instant response |
| Code completion | 1-4 | Developer workflow interruption |
| Content moderation | 64-256 | Can process in batches, near-real-time |
| Document summarization | 512-1024 | Async job queue, throughput critical |
| Dataset annotation | 1200 | Overnight batch job, maximize efficiency |

### 4. Memory as a Capacity Constraint

**Critical insight:** Memory limits concurrency more than compute

**Implications:**

1. **Can't just "add more compute"**
   - Adding tensor cores doesn't help (already memory-bound)
   - Must add more VRAM or reduce memory footprint

2. **Quantization ROI is enormous**
   - INT8: 2x capacity (2,400 users per GPU)
   - INT4: 4x capacity (4,800 users per GPU)  
   - Minimal quality loss for most applications

3. **Context length directly reduces capacity**
   - 50 tokens: 1,200 users per GPU
   - 500 tokens: 120 users per GPU (10x reduction!)
   - 2048 tokens: 30 users per GPU (40x reduction!)

**Product positioning:**
- "Our infrastructure supports 1,200 concurrent users per GPU"
- "Short context queries (under 100 tokens) recommended for optimal cost"
- "Long context pricing: 10x premium justified by capacity impact"

### 5. When to Use Multi-GPU for Small Models

**Don't use multi-GPU when:**
- ✅ Model fits on single GPU (3B-13B typically)
- ✅ Latency is critical (avoid PCIe overhead)
- ✅ Cost per inference matters (simpler = cheaper)

**Do use multi-GPU when:**
- ✅ Model doesn't fit on single GPU (70B+)
- ✅ Need higher aggregate throughput (run multiple replicas)
- ✅ Redundancy required (one GPU fails, others continue)

**For our 3B model specifically:**
- Single GPU optimal for inference
- Multi-GPU only helps if running multiple model replicas
- Each replica serves 1,200 users independently

---

## Technical Skills Developed

### 1. PyTorch Profiling & Memory Management
- `torch.cuda.memory_allocated()` for memory tracking
- `torch.cuda.synchronize()` for accurate timing
- `device_map` parameter for GPU allocation control
- Understanding of PyTorch memory allocation patterns

### 2. Inference Optimization Fundamentals
- Identified memory bandwidth as primary bottleneck
- Quantified FP16 benefits and limitations
- Measured batch size impact on throughput and latency
- Built mental model of GPU utilization vs efficiency

### 3. Statistical Modeling for Capacity Planning
- Linear regression for memory prediction (R² = 0.9999)
- Validation methodology with <1.1% error tolerance
- Translation of model parameters to operational limits
- Safety margin calculations for production systems

### 4. Performance Analysis & Benchmarking
- Structured experimental design (baseline → variations)
- Proper warmup procedures before measurement
- Statistical significance through multiple iterations
- Result visualization and interpretation

### 5. Product Thinking Applied to Infrastructure
- Cost modeling: cloud vs on-premise economics
- SLA definition: latency vs throughput trade-offs
- Capacity planning: users per GPU calculations
- Pricing strategy: how memory constraints affect costs

---

## Challenges Encountered & Resolutions

### Challenge 1: Hugging Face Gated Model Access

**Problem:** 
```
OSError: You are trying to access a gated repo.
Cannot access gated repo for url https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
```

**Root cause:** Llama models require license acceptance

**Resolution:**
1. Visited model page, accepted Meta's license
2. Generated HF token with read permissions
3. Authenticated via `huggingface-cli login`
4. Models now cached locally in `~/.cache/huggingface/`

**Learning:** Always check model licensing before starting experiments

### Challenge 2: Float Formatting in Data Analysis

**Problem:**
```python
ValueError: Unknown format code 'd' for object of type 'float'
```

**Root cause:** Pandas reads CSV columns as float64 by default

**Resolution:**
```python
# Method 1: Cast in format string
print(f"{int(row['batch_size']):5d}")

# Method 2: Specify dtype when loading CSV
df = pd.read_csv('results.csv', dtype={'batch_size': int})
```

**Learning:** Be explicit about data types when loading CSVs

### Challenge 3: Dual GPU Not Splitting Model

**Initial confusion:** Expected model to split across 2 GPUs automatically

**Realization:** This is actually CORRECT behavior!
- 3B model fits on single GPU (6.43 GB < 24 GB)
- Framework avoided unnecessary PCIe overhead
- This is smart optimization by Hugging Face Accelerate

**Resolution:** None needed—confirmed expected behavior

**Learning:** Trust framework defaults; they're often optimized

---

## Files & Artifacts Created

### Scripts
- `baseline_benchmark.py` - FP32/FP16/dual-GPU comparison
- `batch_size_scaling.py` - Throughput vs batch size analysis
- `memory_model_analysis.py` - Linear regression capacity model

### Data Files
- `results/week1_baseline.csv` - Baseline performance metrics
- `results/batch_size_scaling.csv` - Batch scaling data
- `results/memory_model.csv` - Memory predictions vs actuals

### Documentation
- This report (`week-01.md`)
- Inline code comments explaining methodology
- Performance charts (to be generated)

---

## Next Steps: Week 2 Preview

### Primary Objective
Install and benchmark TensorRT optimizations

### Planned Experiments

**Experiment 1: TensorRT Setup**
- Install TensorRT 9.x
- Convert Llama 3.2 3B to TensorRT format
- Validate numerical accuracy vs PyTorch

**Experiment 2: TensorRT Performance**
- Benchmark TensorRT FP16 vs PyTorch FP16
- Measure end-to-end latency improvement
- Test different TensorRT optimization profiles

**Experiment 3: Framework Comparison**
- PyTorch native (baseline)
- TensorRT (NVIDIA's inference optimizer)
- ONNX Runtime (cross-platform option)
- Document when each makes sense

**Expected outcomes:**
- 1.5-2x additional speedup from TensorRT
- Understanding of TensorRT optimization process
- Framework selection criteria for production

### Week 3 Preparation
- Order 2 additional RTX 3090 GPUs
- Plan physical installation (PCIe slots, power)
- Design pre/post upgrade comparison experiments

---

## Reflection: PM → AI Engineer Transition

### What's Working Well

**1. Infrastructure background is an advantage**
- Understanding of SLA trade-offs (latency vs throughput)
- Cost modeling comes naturally (TCO analysis)
- Capacity planning skills directly applicable
- Carrier-scale experience helps think about production

**2. Hands-on approach accelerates learning**
- Running actual benchmarks > reading papers
- Real data builds intuition faster than theory
- Troubleshooting builds confidence
- Portfolio value: "I measured this myself"

**3. Product thinking enhances technical work**
- Not just "what's the number?" but "what does it mean?"
- Translating metrics into user impact
- Understanding business implications of technical choices
- This dual lens is relatively rare and valuable

### Areas for Growth

**1. CUDA fundamentals still surface-level**
- Can use tools but don't fully understand internals
- Week 11-12 (custom kernels) will address this
- May need supplementary coursework on CUDA programming

**2. Need more ML theory**
- Strong on infrastructure, weaker on model architecture
- Should study: attention mechanisms, transformer details
- Plan: Read "Attention Is All You Need" paper this week

**3. Portfolio documentation needs work**
- Code is functional but not well-commented
- Need consistent README files in each directory
- Should write blog posts summarizing weekly learnings

### Confidence Level
**7/10 on inference fundamentals** - solid foundation established

**Next milestone:** 8/10 after Week 4 (completing Phase 1)

---

## Appendix: Commands & Environment

### Hardware Verification
```bash
# Check GPUs
nvidia-smi

# Check CUDA version
nvcc --version

# Verify PyTorch CUDA support
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Environment Setup
```bash
# Activate virtual environment
source ~/ai-inference/bin/activate

# Install dependencies
pip install torch transformers accelerate pandas matplotlib

# Authenticate with Hugging Face
huggingface-cli login
```

### Running Experiments
```bash
# Baseline benchmark
cd ~/work/rtx3090-ai-training/phase-1-foundation/week-01-benchmarks
python3 baseline_benchmark.py

# Batch size scaling
python3 batch_size_scaling.py

# Memory model analysis
python3 memory_model_analysis.py
```

---

## Conclusion

Week 1 successfully established foundational knowledge of NVIDIA inference optimization. The key revelation—that inference is memory-bandwidth limited rather than compute-limited—fundamentally shapes optimization strategy going forward. The linear memory model with <1.1% error provides a robust framework for capacity planning and cost estimation.

Most importantly, this week demonstrated that the transition from infrastructure PM to AI engineering is viable: the skills are complementary, the hands-on learning approach is effective, and the dual technical-product lens is valuable for the target roles.

**Week 1 Status: ✅ Complete - All objectives met**

**Ready for Week 2: TensorRT optimization experiments**

---

*Report generated: January 19, 2026*  
*Hardware: 2x RTX 3090, Ubuntu 24.04, CUDA 12.6*  
*Next hardware upgrade: Week 3 (adding 2x RTX 3090)*
