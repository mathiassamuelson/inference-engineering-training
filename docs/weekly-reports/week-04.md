# Week 4: vLLM — Production Inference Fundamentals

**Duration:** February 2026  
**Hardware:** 4x RTX 3090 (24GB each), Ubuntu 24.04 LTS, CUDA 12.6  
**Test Model:** Llama 3.2 3B Instruct (single GPU, GPU 0)  
**Framework:** vLLM 0.13.0 (V1 engine)

---

## Executive Summary

Week 4 transitioned from Phase 1 baselines to Phase 2 production inference by benchmarking vLLM against the transformers library measurements from Weeks 1 and 3. Four experiments produced five key findings:

1. **vLLM provides ~1.3x throughput improvement from kernel-level optimizations** — CUDA graph capture, Flash Attention, and torch.compile deliver consistent gains at every concurrency level, not just at high load. Single-request throughput improved from 84 to 106 tok/s.

2. **The throughput ceiling moved but didn't shatter** — from ~5,000 to ~6,100 tok/s peak (1.23x). The plateau shape is the same because memory bandwidth remains the fundamental bottleneck for a 3B model on a single GPU. Framework choice provides ~1.3x improvement, not 5-10x.

3. **Continuous batching dramatically improves user experience under mixed workloads** — short requests completed in 0.27s while long requests took 5.35s, a 95% latency reduction versus static batching where all requests would wait for the longest one.

4. **PagedAttention delivers 2.21x memory efficiency** — 112 KB/token vs transformers' 260 KB/token, enabling 2.21x more concurrent requests at every sequence length. This comes from block-level allocation, elimination of fragmentation, and zero padding waste.

5. **Latency SLAs — not throughput or memory — set actual production capacity** — under a real-time chat SLA (p95 < 2s), capacity is ~25 concurrent users, far below the memory-only estimate of 1,200+ users from Week 1.

**Key takeaway:** vLLM's value for small models on a single GPU comes from operational capabilities (graceful degradation, continuous batching, request queuing) more than raw throughput gains. The dramatic improvements will come with larger models, longer sequences, and multi-GPU deployments in Weeks 5-6.

### Correction to Week 1: Grouped Query Attention

Week 4 revealed that the Week 1 per-token KV cache calculation of 344 KB assumed standard Multi-Head Attention (24 KV heads). Llama 3.2 3B actually uses Grouped Query Attention (GQA) with only 8 KV heads (3:1 ratio), giving a correct per-token cost of 112 KB — 3x smaller than originally calculated. The Week 1 measured value of 260 KB/token included both KV cache and framework overhead (activations, attention buffers, memory fragmentation). See the GQA Deep Dive section below.

---

## Objectives

- ✅ Install and verify vLLM with Llama 3.2 3B
- ✅ Benchmark single-GPU throughput against Week 1 transformers baselines
- ✅ Simulate concurrent users with realistic traffic patterns
- ✅ Analyze PagedAttention memory efficiency vs transformers
- ✅ Establish production capacity estimates under SLA constraints
- ✅ Understand GQA's impact on KV cache sizing

---

## Experiment 1: vLLM Installation & Verification

### Setup

vLLM 0.13.0 was already installed in the `ai-inference` environment. The server was launched on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --dtype float16 --max-model-len 4096 --port 8000 --disable-log-requests
```

### Startup Observations

| Metric | Value | Notes |
|--------|-------|-------|
| Engine version | V1 (vLLM 0.13.0) | Latest engine architecture |
| Model memory | 6.016 GiB | Nearly identical to transformers' 6.43 GB |
| KV cache pool | 13.97 GiB | Pre-allocated from remaining GPU memory |
| KV cache capacity | 130,752 tokens | At 112 KB/token (GQA-corrected) |
| Max concurrency (4,096 tokens) | 31.9x | Pool capacity ÷ max sequence length |
| Attention backend | FLASH_ATTN | Memory-efficient attention computation |
| torch.compile | 4.94s one-time cost | Fused CUDA kernels compiled at startup |
| CUDA graphs captured | 86 total | 51 piecewise + 35 full, batch sizes 1-512 |
| Chunked prefill | Enabled (2,048 max) | Prevents long prompts from blocking short ones |
| Model load time | 1.6s | Weights loaded from cached safetensors |

### Six Production Mechanisms Identified at Startup

**1. PagedAttention (KV cache pool):** Pre-allocated 14 GB as a managed pool of fixed-size blocks (16 tokens/block), allocated incrementally as sequences grow. No fragmentation, no padding waste.

**2. CUDA graph capture:** Pre-compiled 86 execution graphs for batch sizes 1-512, eliminating kernel launch overhead during inference. Each graph is a frozen sequence of GPU operations that can be replayed without CPU involvement.

**3. Fused kernels (torch.compile):** One-time compilation of optimized kernels that combine multiple operations (e.g., attention + softmax + value projection) into single GPU launches, reducing memory traffic.

**4. Flash Attention:** Computes attention in tiled blocks that fit in GPU SRAM rather than materializing the full `[seq_len × seq_len]` attention matrix in VRAM. Reduces attention memory from O(n²) to O(1).

**5. Chunked prefill:** Long prompts are broken into 2,048-token chunks and interleaved with decode steps from other requests, preventing prompt processing from blocking active generations.

**6. Continuous batching:** Requests enter and exit the processing batch dynamically as they arrive and complete, rather than waiting for all requests in a static batch to finish.

---

## Experiment 2: Single-GPU Throughput — vLLM vs Transformers

### Setup

Concurrent HTTP requests to the vLLM server at concurrency levels matching Week 1 batch sizes. Same model (Llama 3.2 3B), same generation length (50 tokens), same GPU.

### Results

| Concurrency | vLLM Total | TF Total | Speedup | vLLM/Sample | TF/Sample | Speedup | Mean Latency | P95 Latency |
|-------------|-----------|----------|---------|-------------|-----------|---------|-------------|-------------|
| 1 | 106 tok/s | 84 tok/s | 1.26x | 106.2 tok/s | 84.2 tok/s | 1.26x | 0.471s | 0.471s |
| 8 | 772 tok/s | 609 tok/s | 1.27x | 96.5 tok/s | 76.1 tok/s | 1.27x | 0.517s | 0.518s |
| 64 | 4,348 tok/s | 3,376 tok/s | 1.29x | 67.9 tok/s | 52.7 tok/s | 1.29x | 0.732s | 0.734s |
| 128 | 5,823 tok/s | 4,291 tok/s | 1.36x | 45.5 tok/s | 33.5 tok/s | 1.36x | 1.094s | 1.096s |
| 256 | 6,142 tok/s | 4,656 tok/s | 1.32x | 24.0 tok/s | 18.2 tok/s | 1.32x | 2.070s | 2.079s |
| 512 | 6,142 tok/s | 4,703 tok/s | 1.31x | 12.0 tok/s | 9.2 tok/s | 1.30x | 3.165s | 4.153s |
| 1024 | 6,061 tok/s | 4,951 tok/s | 1.22x | 6.0 tok/s | 4.8 tok/s | 1.24x | 5.314s | 8.367s |
| 1200 | 6,028 tok/s | 4,998 tok/s | 1.21x | 5.9 tok/s | 4.2 tok/s | 1.41x | 5.345s | 8.408s |

### Key Findings

**1. Consistent ~1.3x Improvement from Kernel Efficiency**

The speedup ratio is roughly constant (1.2-1.4x) across all concurrency levels. This means the improvement comes from per-request kernel optimizations (CUDA graphs, Flash Attention, compiled kernels), not from continuous batching. With identical prompt and generation lengths sent simultaneously, continuous batching has nothing to differentiate — all requests look the same.

**2. Throughput Ceiling Moved from ~5,000 to ~6,100 tok/s**

The plateau shape is identical to Week 1: throughput climbs through concurrency 256, then flattens. Memory bandwidth remains the fundamental limit for a 3B model — even perfect software can't read 6 GB of weights faster than 936 GB/s allows.

**3. Per-Sample Degradation Pattern Persists**

With transformers: 84 → 4.2 tok/s (95% collapse). With vLLM: 106 → 5.9 tok/s (94.4% collapse). Nearly identical degradation curves confirm this is hardware bandwidth saturation, not a software problem vLLM can fix with better batching.

**4. Zero Failures at Any Concurrency**

vLLM successfully handled all 1,200 concurrent requests with request queuing and backpressure. Transformers would have either OOM'd or required manual batch management. This operational resilience is a production-readiness differentiator independent of raw throughput.

---

## Experiment 3: Concurrent User Simulation

### Setup

Realistic production traffic simulation: staggered arrivals, variable prompt lengths, variable generation lengths (5-200 tokens), sequential requests per user (5 per user). Workload mix: 30% short (20-40 tokens), 40% medium (50-100 tokens), 20% long (100-200 tokens), 10% tiny (5-10 tokens).

### Results

| Users | System tok/s | User tok/s (mean) | User tok/s (min) | Lat p50 | Lat p95 | Lat p99 | Lat max | Avg Gen | Failures |
|-------|-------------|-------------------|------------------|---------|---------|---------|---------|---------|----------|
| 1 | 107 | 106.8 | 106.7 | 1.432s | 1.756s | 1.756s | 1.756s | 142.0t | 0 |
| 5 | 358 | 98.4 | 87.1 | 0.668s | 1.797s | 1.912s | 1.912s | 73.4t | 0 |
| 10 | 668 | 95.6 | 84.3 | 0.544s | 1.833s | 2.000s | 2.000s | 69.7t | 0 |
| 25 | 1,373 | 86.5 | 72.9 | 0.630s | 1.958s | 2.270s | 2.311s | 68.9t | 0 |
| 50 | 2,435 | 73.9 | 53.8 | 0.911s | 2.406s | 2.645s | 2.744s | 76.5t | 0 |
| 100 | 3,436 | 51.3 | 30.6 | 1.292s | 3.486s | 4.092s | 4.208s | 74.2t | 0 |
| 150 | 3,813 | 36.6 | 20.9 | 1.668s | 4.797s | 6.019s | 6.346s | 67.0t | 0 |
| 200 | 4,121 | 30.0 | 17.5 | 2.048s | 6.489s | 7.813s | 8.177s | 68.9t | 0 |

### Production Capacity Under SLA Constraints

| SLA Target | Max Users | System Throughput | Per-User tok/s |
|------------|-----------|-------------------|----------------|
| Real-time chat (p95 < 2s) | 25 | 1,373 tok/s | 87 tok/s |
| API serving (p95 < 5s) | 150 | 3,813 tok/s | 37 tok/s |
| Batch processing (p95 < 10s) | 200+ | 4,121 tok/s | 30 tok/s |

### Key Findings

**1. Graceful Degradation Under Load**

Per-user throughput declined smoothly from 107 to 30 tok/s across 1-200 users — no cliff or catastrophic collapse. At 100 users, each user still received 51 tok/s, which is usable for interactive applications. Compare to transformers where per-sample throughput collapsed 95% at high batch sizes.

**2. Latency SLAs Are the Binding Constraint**

Week 1 estimated 100-150 users per GPU based on throughput analysis. Under realistic traffic with a real-time chat SLA (p95 < 2s), actual capacity is only ~25 users. Memory-only capacity estimates (1,200+ users) are even more misleading. Production planning must work backward from latency requirements, not forward from memory or throughput ceilings.

**3. System Throughput Peaked at ~4,100 tok/s**

Lower than Experiment 2's 6,100 tok/s because the effective in-flight concurrency equals the user count (each user waits for one response before sending the next), and variable generation lengths (avg ~70 tokens vs fixed 50) change the workload profile. This is more realistic than the simultaneous-launch pattern.

**4. Zero Failures Across All Levels**

200 concurrent users with no request failures, no OOM errors, no timeout-related drops. vLLM's request queuing and backpressure handled overload gracefully.

---

## Experiment 4: PagedAttention Memory Analysis

### Part A: Per-Token Memory Cost Comparison

| Source | Per-Token KV Cost | Notes |
|--------|-------------------|-------|
| Theoretical (MHA, Week 1) | 0.328 MB (328 KB) | Incorrect: assumed 24 KV heads |
| Theoretical (GQA, corrected) | 0.112 MB (112 KB) | Correct: 8 KV heads |
| Transformers (Week 1 measured) | 0.261 MB (261 KB) | Includes framework overhead |
| vLLM (pool ÷ capacity) | 0.109 MB (109 KB) | Pure KV cache, block-managed |

**vLLM uses 0.42x the memory per token compared to transformers** (109 vs 261 KB). This gap comes from two sources:

1. **No framework overhead:** Transformers' 261 KB/token includes activation tensors, attention score matrices, autograd tracking, and memory fragmentation. vLLM computes activations in-place, discards attention scores within fused kernels, and eliminates fragmentation via block allocation.

2. **Block-level allocation:** PagedAttention allocates in 16-token blocks, only as needed. No pre-allocation of maximum sequence length, no padding waste for variable-length sequences.

### Part B: Capacity at Different Sequence Lengths

| Seq Length | vLLM Max Concurrent | Transformers Max Batch | TF Memory/Request | vLLM Advantage |
|------------|--------------------|-----------------------|-------------------|----------------|
| 50 | 2,615 | 1,181 | 13.0 MB | 2.21x |
| 100 | 1,308 | 591 | 26.1 MB | 2.21x |
| 256 | 511 | 231 | 66.7 MB | 2.21x |
| 500 | 262 | 118 | 130.3 MB | 2.21x |
| 1,024 | 128 | 58 | 266.9 MB | 2.21x |
| 2,048 | 64 | 29 | 533.7 MB | 2.21x |
| 4,096 | 32 | 14 | 1,067 MB | 2.21x |

The constant 2.21x advantage at every sequence length reflects the per-token efficiency ratio applied uniformly. At short sequences (50 tokens), both frameworks handle far more concurrent requests than SLA constraints allow. At long sequences (4,096 tokens), the difference is 32 vs 14 concurrent requests — the gap between serving a production workload and not.

### Part C: Live KV Cache Utilization

KV cache utilization metrics reported 0% across all scenarios due to a metrics endpoint incompatibility with vLLM V1 engine's metric naming conventions. All requests succeeded with expected throughput, confirming the KV cache was functional — the observability gap is in metrics scraping, not cache behavior. This will be revisited in Week 5-6 with updated metrics parsing.

### Part D: Mixed Sequence Length Efficiency — Continuous Batching

30 concurrent requests: 10 short (20 tokens), 10 medium (200 tokens), 10 long (500 tokens).

| Request Type | Count | Avg Generated | Per-Request tok/s | Lat p50 | Lat p95 |
|-------------|-------|--------------|-------------------|---------|---------|
| Short (≤25 tokens) | 10 | 20.0 tokens | 74.3 | 0.273s | 0.273s |
| Medium (26-250 tokens) | 10 | 200.0 tokens | 93.3 | 2.145s | 2.145s |
| Long (>250 tokens) | 10 | 500.0 tokens | 93.5 | 5.347s | 5.347s |

**Continuous batching demonstrated clearly:** Short requests completed in 0.27s while long requests took 5.35s. Without continuous batching (static batching), all 30 requests would have been blocked until the 500-token requests finished — every short request would have waited 5.35s instead of 0.27s.

**Continuous batching saved short requests 5.08s (95% latency reduction).**

Medium and long requests both achieved ~93 tok/s — faster than the transformers baseline of 84 tok/s at batch=1. Short requests show lower apparent tok/s (74.3) because the 0.27s latency includes HTTP and scheduling overhead that dominates when only generating 20 tokens.

---

## GQA Deep Dive: Correcting the Week 1 KV Cache Calculation

### The Problem with the Week 1 Calculation

Week 1 calculated per-token KV cache as:

```
2 (K and V) × 28 layers × 24 heads × 128 dim × 2 bytes = 344,064 bytes ≈ 0.33 MB per token
```

This assumed all 24 attention heads are KV heads (Multi-Head Attention). Llama 3.2 3B actually uses **Grouped Query Attention (GQA)**, where 24 query heads share only 8 KV heads.

### The Correction

```
2 (K and V) × 28 layers × 8 KV heads × 128 dim × 2 bytes = 114,688 bytes ≈ 0.112 MB per token
```

This matches vLLM's pre-allocated pool capacity: 13.97 GiB ÷ 112 KB/token ≈ 130,752 tokens (with block alignment overhead accounting for the small difference from the theoretical 137,079).

### What Is GQA?

Three attention architectures differ in how many KV heads exist relative to query heads:

**Multi-Head Attention (MHA)** — the original design:
- 24 query heads, 24 KV heads (1:1 ratio)
- KV cache per token: 344 KB
- Maximum representational capacity, maximum memory cost

**Multi-Query Attention (MQA)** — the opposite extreme:
- 24 query heads, 1 KV head (24:1 ratio)
- KV cache per token: 14.3 KB
- Massive savings, but quality degrades — all heads forced to agree on relevance

**Grouped Query Attention (GQA)** — the practical middle ground:
- 24 query heads, 8 KV heads (3:1 ratio)
- KV cache per token: 112 KB
- Preserves most MHA quality while capturing most MQA memory savings

### How GQA Works Mechanically

In standard MHA, each head has dedicated K and V projections:

```
Head 0: Q₀ = x × Wq₀, K₀ = x × Wk₀, V₀ = x × Wv₀
Head 1: Q₁ = x × Wq₁, K₁ = x × Wk₁, V₁ = x × Wv₁
...24 independent sets of Q, K, V
```

In GQA with 8 KV groups, heads 0-2 share a KV head:

```
Head 0: Q₀ = x × Wq₀  → uses K_group0, V_group0
Head 1: Q₁ = x × Wq₁  → uses K_group0, V_group0  (same K,V)
Head 2: Q₂ = x × Wq₂  → uses K_group0, V_group0  (same K,V)
Head 3: Q₃ = x × Wq₃  → uses K_group1, V_group1
...
```

Each query head retains its own unique Q projection — they "ask different questions." But within a group, they search the same keys and retrieve the same values. Research found that adjacent MHA heads tend to converge on similar K/V representations anyway, so sharing them explicitly costs minimal quality.

### Weight Matrix Implications

GQA also reduces attention projection parameters:

```
MHA:  24 Wq + 24 Wk + 24 Wv = 72 projection matrices per layer
GQA:  24 Wq +  8 Wk +  8 Wv = 40 projection matrices per layer (44% fewer)
```

The saved parameters are reallocated elsewhere (typically larger MLP layers), keeping the total parameter count at 3B while optimizing the memory-intensive KV cache path.

### Architecture Comparison (Corrected)

| Model | Attention Type | Query Heads | KV Heads | Ratio | KV/Token (FP16) |
|-------|---------------|-------------|----------|-------|-----------------|
| GPT-2 | MHA | 12 | 12 | 1:1 | 38 KB |
| Llama 3.2 1B | GQA | 32 | 8 | 4:1 | 32 KB |
| **Llama 3.2 3B** | **GQA** | **24** | **8** | **3:1** | **112 KB** |
| Llama 3.1 8B | GQA | 32 | 8 | 4:1 | 131 KB |
| Llama 3.1 70B | GQA | 64 | 8 | 8:1 | 328 KB |
| Mistral 7B | GQA | 32 | 8 | 4:1 | 131 KB |

Llama 3.1 70B uses an 8:1 ratio — 64 query heads sharing only 8 KV heads. Without GQA, its KV cache would be 8x larger (2.6 MB/token), making production serving essentially unviable. GQA is what makes large model deployment practical.

### Where the Week 1 Gap Came From

| Source | Per-Token Cost | What's Included |
|--------|---------------|-----------------|
| MHA theoretical (Week 1) | 344 KB | Wrong head count assumption |
| GQA theoretical (corrected) | 112 KB | Pure KV cache |
| Transformers measured (Week 1) | 261 KB | KV cache + activations + attention buffers + fragmentation |
| vLLM measured (Week 4) | 109 KB | Pure KV cache, block-managed |

Transformers' 261 KB/token exceeded even the correct 112 KB because it includes per-request activation tensors, full attention score matrices, Python/autograd tracking overhead, and memory fragmentation from individual allocations. vLLM eliminates all of this through in-place computation, fused kernels, and block-level memory management.

---

## Product & Engineering Insights

### 1. Framework Value Proposition Depends on Workload Scale

| Scenario | vLLM Advantage | Why |
|----------|---------------|-----|
| Single request, small model | ~1.3x throughput | Kernel optimizations only |
| High concurrency, small model | ~1.3x throughput + operational resilience | Same kernels + request management |
| Mixed workloads | 95% latency reduction for short requests | Continuous batching |
| Long context (4K+ tokens) | 2.21x memory capacity | PagedAttention efficiency |
| Large models (70B+) | Essential | Memory management at scale |

For a 3B model on a single GPU, vLLM's value is primarily operational (graceful degradation, zero failures, continuous batching) rather than raw throughput. The throughput ceiling is hardware-bound at ~6,100 tok/s regardless of framework.

### 2. Capacity Planning Must Start from SLAs

| Planning Approach | Estimated Capacity | Reality |
|-------------------|-------------------|---------|
| Memory-only (Week 1) | 1,200 users/GPU | Misleading |
| Throughput-limited (Week 1) | 100-150 users/GPU | Optimistic |
| SLA-constrained (Week 4) | 25 users (chat) / 150 users (API) | Correct |

The gap between memory-based estimates and SLA-constrained capacity is 50-100x. Any production capacity model that doesn't start from latency requirements will be dramatically wrong.

### 3. Updated Cost Analysis

**Single RTX 3090 with vLLM (Llama 3.2 3B):**

| SLA | Users/GPU | 4-GPU System | Cloud Equivalent |
|-----|-----------|-------------|------------------|
| Real-time chat (p95 < 2s) | 25 | 100 users | ~$2,500/month |
| API serving (p95 < 5s) | 150 | 600 users | ~$4,000/month |

On-premise (4x RTX 3090 at ~$6,000 upfront) breaks even in 2-3 months vs cloud for sustained workloads.

### 4. When vLLM's Advantages Compound

The 1.3x throughput improvement and 2.21x memory efficiency measured here are for the simplest case: a small model, short sequences, single GPU. These advantages compound when:

- **Models get larger:** Memory management becomes critical for 8B+ models where KV cache competes with model weights for VRAM
- **Sequences get longer:** PagedAttention's block allocation prevents the fragmentation that makes transformers unusable at 4K+ contexts
- **Traffic is variable:** Continuous batching prevents short requests from being blocked by long ones — the 95% latency reduction in Experiment 4 is the clearest demonstration
- **Multi-GPU deployments:** vLLM handles tensor parallelism and data parallelism internally, eliminating the manual orchestration required in Week 3

---

## Technical Skills Developed

1. **vLLM server deployment:** Configuration, API endpoints, startup diagnostics
2. **Async load testing:** aiohttp-based concurrent request benchmarking
3. **Production traffic simulation:** Variable workloads, staggered arrivals, sequential user sessions
4. **SLA-based capacity planning:** Working backward from latency requirements
5. **Memory architecture analysis:** GQA's impact on KV cache sizing and serving economics
6. **Framework comparison methodology:** Controlled experiments isolating kernel efficiency from batching effects

---

## Challenges & Resolutions

### 1. Benchmark Script Failures (All Requests Failed)
**Problem:** Initial Experiment 2 run showed all requests failing across every concurrency level  
**Root cause:** vLLM server had been shut down between experiments  
**Resolution:** Restarted server, confirmed with curl health check before re-running

### 2. vLLM V1 Metrics Incompatibility
**Problem:** Part C KV cache utilization reported 0% for all scenarios  
**Root cause:** vLLM 0.13.0's V1 engine exposes metrics under different names/labels than expected  
**Resolution:** Deferred to Week 5-6; calculated values from Parts A and B provide the needed analysis

### 3. Default Sampling Parameters Override
**Problem:** vLLM overrode temperature=0.0 with model defaults (temperature=0.6, top_p=0.9)  
**Resolution:** Explicit temperature=0.0 in request payloads overrides the server defaults; confirmed deterministic outputs

---

## Key Learnings: Theory vs Practice

### What I Expected
- vLLM would break the 5,000 tok/s ceiling by 3-5x
- PagedAttention would show dramatic memory savings at short sequences
- Continuous batching would improve total throughput significantly
- Single-request latency would be similar or slightly worse (server overhead)

### What I Measured
- Throughput ceiling moved from 5,000 to 6,100 tok/s (1.23x, not 3-5x)
- PagedAttention provides consistent 2.21x advantage at all sequence lengths
- Continuous batching improves user experience (95% latency reduction) not total throughput
- Single-request throughput improved 1.26x (106 vs 84 tok/s) from kernel optimizations

### Critical Insight

**vLLM's value scales with system complexity.** On a single GPU with a small model and short sequences, the advantage is modest (~1.3x throughput, 2.2x memory). The real value emerges at scale: larger models where memory management is critical, mixed workloads where continuous batching prevents head-of-line blocking, and multi-GPU deployments where internal parallelism eliminates manual orchestration. Week 4 establishes the baseline; Weeks 5-6 will test the scaling hypothesis.

---

## Interview Articulations

### vLLM vs Transformers Performance

"We benchmarked vLLM against the transformers library on Llama 3.2 3B across the same concurrency levels. vLLM delivered a consistent ~1.3x throughput improvement from kernel-level optimizations — CUDA graph capture, Flash Attention, and torch.compile — independent of concurrency level. The throughput ceiling moved from 5,000 to 6,100 tok/s but maintained the same plateau shape, confirming that memory bandwidth is the fundamental bottleneck for small models on a single GPU. The key insight: framework choice provides ~1.3x improvement for small models, but the operational advantages — zero failures at 1,200 concurrent requests, graceful degradation, continuous batching — are the real production differentiators."

### Continuous Batching

"Continuous batching's value is latency management, not throughput improvement. In our mixed-workload test, short requests (20 tokens) completed in 0.27 seconds while long requests (500 tokens) took 5.35 seconds — running simultaneously. Without continuous batching, static batching would have blocked every request until the longest one finished, making all 30 requests take 5.35 seconds. That's a 95% latency reduction for short requests. In production, this means a classification query doesn't wait behind a long-form generation, which directly impacts SLA compliance and user experience."

### PagedAttention Memory Efficiency

"PagedAttention delivered 2.21x memory efficiency versus the transformers library — 109 KB/token vs 261 KB/token. The savings come from three mechanisms: block-level allocation that eliminates fragmentation, incremental allocation that avoids pre-reserving maximum sequence length, and the elimination of per-request overhead that transformers carries for activations, attention score matrices, and Python-level memory management. At 4,096-token context lengths, this translates to 32 vs 14 maximum concurrent requests — the difference between a viable production deployment and one that can't meet capacity requirements."

### SLA-Driven Capacity Planning

"Week 4 showed that latency SLAs — not memory or throughput — are the binding constraint for production capacity. Memory analysis suggested 1,200 users per GPU, throughput analysis said 100-150, but under a real-time chat SLA of p95 under 2 seconds, actual capacity was only 25 users. That's a 50x gap between the naive estimate and reality. Production planning must work backward from latency requirements: define the SLA, measure capacity at that constraint, then size infrastructure. Any approach that starts from hardware specs and works forward will dramatically overestimate capacity."

### Grouped Query Attention

"Modern LLMs use Grouped Query Attention to reduce KV cache memory by sharing Key and Value heads across multiple Query heads. Llama 3.2 3B uses a 3:1 ratio — 24 query heads share 8 KV heads — reducing per-token cache from 344 KB to 112 KB. This 3x reduction translates directly to 3x more concurrent requests on the same GPU. The insight is that adjacent attention heads tend to learn similar K/V representations during training, so sharing them explicitly costs minimal quality while dramatically improving serving economics. Larger models use even higher ratios — Llama 3.1 70B uses 8:1 — which is what makes 70B-plus model serving feasible at all."

### Framework Selection for Production

"Framework choice provides modest throughput gains for small models on single GPUs — about 1.3x in our testing — but the advantages compound with system complexity. vLLM's real value is threefold: operational resilience through request queuing and graceful degradation, user experience through continuous batching that prevents head-of-line blocking, and memory efficiency through PagedAttention that becomes critical as models and context lengths scale. For a 3B model at 50 tokens, these are nice-to-haves. For a 70B model at 4,096 tokens with variable traffic, they're requirements."

---

## Files Created

**Scripts:**
- `scripts/vllm_throughput_benchmark.py` — Experiment 2: concurrent request throughput vs Week 1
- `scripts/vllm_concurrent_users.py` — Experiment 3: realistic traffic simulation
- `scripts/vllm_memory_analysis.py` — Experiment 4: PagedAttention memory comparison

**Documentation:**
- `week-04.md` (this report)

---

## Conclusion

Week 4 established vLLM as the production inference framework and revealed that its value for small models is primarily operational rather than throughput-based. The ~1.3x kernel-level improvement is real but modest; the transformative capabilities — continuous batching, PagedAttention memory management, graceful degradation — become visible in mixed workloads and will become essential as model size, context length, and deployment complexity increase in Weeks 5-6.

The GQA correction to Week 1's KV cache calculation is a significant finding: understanding the actual 112 KB/token cost (not 344 KB) is essential for accurate capacity planning and directly explains vLLM's startup configuration. This correction illustrates a broader principle — architectural details that seem academic (attention head sharing) have direct, measurable impact on production serving costs.

The most important insight is the SLA-driven capacity planning framework. Memory says 1,200 users, throughput says 100-150, SLAs say 25. Production decisions must start from the SLA constraint and work backward.

**Week 4 Status:** ✅ Complete — All objectives met, Phase 2 foundation established

**Ready for Weeks 5-6:** vLLM multi-GPU, larger models, sustained load testing

---

*Report generated: February 2026*  
*Hardware: 4x RTX 3090, Gigabyte B650 Eagle AX, Ubuntu 24.04, CUDA 12.6*  
*Framework: vLLM 0.13.0 (V1 engine)*  
*Model: Llama 3.2 3B Instruct (single GPU)*