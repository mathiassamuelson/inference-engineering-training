# Week 6 Report: Larger Model Scaling & Triton Introduction

**Training Program:** RTX 3090 AI Infrastructure — Phase 2: Production Inference at Scale  
**Week:** 6 of 24  
**Date:** February 2026  
**Hardware:** 4x RTX 3090 (96GB total), Gigabyte B650 Eagle AX, Ubuntu 24.04, CUDA 12.6  
**Frameworks:** vLLM 0.13.0 (V1 engine), NVIDIA Triton Inference Server, ONNX Runtime

---

## Executive Summary

Week 6 answered the central question from Week 5: "At what model size does vLLM's advantage grow beyond 7x?" The answer is that it doesn't — the advantage belongs to smaller models that can serve more efficiently on fewer GPUs. Scaling from 3B to 14B parameters across two GPUs via tensor parallelism exposed a stark throughput cliff: Qwen 2.5 14B on 2 GPUs achieves only 316.5 tok/s peak throughput, compared to 3,115.9 tok/s for Mistral 7B on a single GPU. Per-GPU efficiency drops by 20x, and cost per token rises by nearly 20x.

This week also established the first non-LLM model serving capability through Triton Inference Server, deploying an embedding model with dynamic batching and Prometheus metrics — laying the groundwork for multi-model orchestration in Week 7.

**Key numbers:** 38.6 tok/s single-request for 14B (vs 53.1 for 7B, vs 106 for 3B), 19.69x cost per token penalty for 14B over 7B, and 23.2 req/s peak throughput for Triton embedding serving with 3.5x dynamic batching speedup.

---

## Objectives

1. Scale to a larger model (Qwen 2.5 14B) across 2 GPUs using tensor parallelism (TP=2)
2. Compare single-GPU 7B serving vs multi-GPU 14B serving across throughput, latency, and cost
3. Deploy a non-LLM model on Triton Inference Server with dynamic batching and production metrics

**Status:** ✅ All objectives met

---

## Experiment 1: Qwen 2.5 14B Tensor Parallel Benchmark

**Setup:** Qwen/Qwen2.5-14B-Instruct served via vLLM with tensor parallelism degree 2 (TP=2) across 2x RTX 3090 GPUs. 50 tokens generated per request. Concurrency sweep from 1 to 256.

### Throughput Results

| Concurrency | System tok/s | Per-Sample tok/s | Mean Latency | p95 Latency |
|-------------|-------------|------------------|-------------|------------|
| 1 | 38.6 | 38.6 | 1.295s | 1.295s |
| 2 | 68.1 | 34.1 | 1.455s | 1.468s |
| 4 | 112.0 | 28.0 | 1.777s | 1.785s |
| 8 | 165.1 | 20.6 | 2.417s | 2.423s |
| 16 | 218.3 | 13.6 | 3.660s | 3.665s |
| 32 | 267.3 | 8.4 | 5.982s | 5.986s |
| 64 | 303.6 | 4.7 | 10.536s | 10.539s |
| 128 | 312.1 | 2.4 | 20.498s | 20.502s |
| 256 | 316.5 | 1.2 | 40.280s | 40.437s |

### Cross-Week Comparison

| Metric | Week 1 (3B, transformers) | Week 4 (3B, vLLM 1 GPU) | Week 6 (14B, vLLM TP=2) |
|--------|--------------------------|--------------------------|--------------------------|
| Single request | 84 tok/s | 106 tok/s | 38.6 tok/s |
| Per-token latency | 11.9 ms | 9.4 ms | 25.9 ms |
| Peak throughput | ~5,000 tok/s | 4,731 tok/s | 316.5 tok/s |

### KV Cache Analysis

| Model | KV Heads | KV per Token | Pool Size | Token Capacity |
|-------|----------|-------------|-----------|----------------|
| Llama 3.2 3B | 8 | 112 KB | 13.97 GiB | ~130,000 |
| Qwen 2.5 14B | 8 | 192 KB | 5.81 GiB | 63,440 |

Qwen 2.5 14B uses Grouped Query Attention with 40 query heads and 8 KV heads (5:1 ratio), producing 5,120 hidden dimensions across 48 layers. Each KV head has 128 dimensions, yielding 192 KB per token in the KV cache — 1.71x more per token than Llama 3.2 3B. But the real constraint is that after loading model weights across 2 GPUs with TP=2, only 5.81 GiB remains for the KV cache pool, giving 2.05x less total token capacity than the 3B model on a single GPU.

### Key Findings

The 14B model hits a throughput ceiling around 316 tok/s regardless of concurrency — the system is saturated at concurrency 64 with only marginal gains beyond that point. This contrasts sharply with smaller models where vLLM continues scaling throughput up through high concurrency levels.

The tensor parallel communication overhead is significant. With TP=2, every token generation requires all-reduce synchronization between 2 GPUs for each of the 48 transformer layers. On this hardware's PCIe topology, that synchronization adds measurable latency per token. The 38.6 tok/s single-request rate for 14B versus 53.1 tok/s for 7B on one GPU shows a 1.38x per-request penalty that compounds under load — widening to 9.84x at peak concurrency.

Per-sample throughput degrades severely under load: from 38.6 tok/s at concurrency 1 to just 1.2 tok/s at concurrency 256 — a 97% drop. The 14B model maintains ≥20 tok/s per sample only up to concurrency 8, compared to Mistral 7B which sustains ≥20 tok/s through concurrency 128.

---

## Experiment 2: Mistral 7B Baseline & Economics Comparison

**Setup:** Mistral-7B-Instruct-v0.3 served on a single RTX 3090 (GPU 0) via vLLM, same concurrency sweep and generation parameters as Experiment 1.

### Throughput Results

| Concurrency | System tok/s | Per-Sample tok/s | Mean Latency | p95 Latency |
|-------------|-------------|------------------|-------------|------------|
| 1 | 53.1 | 53.1 | 0.941s | 0.941s |
| 2 | 100.7 | 50.4 | 0.983s | 0.992s |
| 4 | 199.7 | 49.9 | 0.996s | 1.001s |
| 8 | 391.0 | 48.9 | 1.020s | 1.023s |
| 16 | 710.3 | 44.4 | 1.124s | 1.126s |
| 32 | 1,356.6 | 42.4 | 1.177s | 1.179s |
| 64 | 2,229.1 | 34.8 | 1.434s | 1.435s |
| 128 | 3,014.2 | 23.5 | 2.120s | 2.122s |
| 256 | 3,115.9 | 12.2 | 4.086s | 4.102s |

### Comparison with Week 5 Mistral 7B Results

Week 5 measured Mistral 7B at 53.2 tok/s single-request and 2,478.6 tok/s peak at concurrency 64. This week's results are consistent: 53.1 tok/s single-request and 2,229.1 tok/s at concurrency 64. The slight variance falls within normal measurement range. The new test extended to concurrency 256, revealing that the 7B model continues gaining throughput (3,115.9 tok/s) well beyond where the 14B model plateaus.

### Cost Efficiency Analysis

| Metric | Mistral 7B (1 GPU) | Qwen 14B (2 GPUs) |
|--------|--------------------|--------------------|
| GPUs used | 1 | 2 |
| Single-request tok/s | 53.1 | 38.6 |
| Peak throughput | 3,115.9 tok/s | 316.5 tok/s |
| Per-GPU peak throughput | 3,115.9 tok/s | 158.2 tok/s |
| Cloud cost assumption | $1.00/hr | $2.00/hr |
| Tokens per $1 (peak) | 11,217,374 | 569,700 |
| Cost ratio per token | 1.00x (baseline) | 19.69x |

### SLA-Driven Capacity

| p95 Latency Target | 7B Concurrent Users | 7B Throughput | 14B Concurrent Users | 14B Throughput |
|--------------------|--------------------:|-------------:|---------------------:|-------------:|
| < 1.0s | 2 | 100.7 tok/s | 0 | — |
| < 2.0s | 64 | 2,229.1 tok/s | 4 | 112.0 tok/s |
| < 5.0s | 256 | 3,115.9 tok/s | 16 | 218.3 tok/s |
| < 10.0s | 256 | 3,115.9 tok/s | 32 | 267.3 tok/s |

Under a typical interactive SLA of p95 < 2s, the 7B model serves 16x more concurrent users at 20x the throughput, using half the GPU resources. The 14B model can't meet a 1-second p95 target at any concurrency level.

### Decision Framework

**Use 7B when:** latency-sensitive, high-throughput, cost-constrained, or the task is well within the model's capability (summarization, extraction, classification, structured output).

**Use 14B when:** quality is critical and wrong answers carry significant downstream cost, reasoning depth matters, the task demands broader knowledge or more nuanced responses, and latency requirements are relaxed (p95 > 5s acceptable).

The 14B model is not a universal upgrade over 7B — it's a targeted tool for quality-critical workloads where the 19.69x cost premium is justified by the value of better outputs.

---

## Experiment 3: Triton Inference Server — Embedding Model

**Setup:** NVIDIA Triton Inference Server deployed with all-MiniLM-L6-v2 (22M parameters, 384-dimension embeddings) converted to ONNX format. Hosted on GPU 2 with ONNX Runtime backend, dynamic batching configured with preferred batch sizes [8, 16, 32] and 100ms max queue delay.

### Server Configuration

The model repository follows Triton's required directory structure with a `config.pbtxt` defining input/output tensors, instance groups, and dynamic batching parameters. The model accepts tokenized inputs (input_ids, attention_mask, token_type_ids) and returns the last hidden state tensor, from which CLS embeddings are extracted.

### Functional Validation

Semantic similarity testing confirmed correct model behavior. GPU-related sentences scored high similarity (0.728 for "GPU memory bandwidth limits inference speed" ↔ "throughput of neural network inference depends on VRAM"), while an unrelated sentence about cooking scored low (0.492). CUDA-related content scored as expected medium similarity (0.659).

### Dynamic Batching Throughput

| Concurrency | Requests/sec | Mean Latency | p95 Latency |
|-------------|-------------|-------------|------------|
| 1 | 6.5 | 146.6ms | 146.6ms |
| 2 | 10.2 | 168.0ms | 188.9ms |
| 4 | 14.1 | 212.0ms | 276.0ms |
| 8 | 22.5 | 198.5ms | 347.1ms |
| 16 | 22.9 | 369.5ms | 690.8ms |
| 32 | 23.1 | 707.9ms | 1,308.2ms |
| 64 | 23.0 | 1,414.5ms | 2,643.9ms |
| 128 | 23.2 | 2,782.6ms | 5,244.2ms |

Peak throughput reached 23.2 req/s at concurrency 128 — a 3.5x improvement over the single-request rate of 6.5 req/s, driven entirely by Triton's dynamic batching. The throughput curve plateaus at concurrency 8, suggesting the GPU saturates quickly for this small model. Mean latency at the plateau (198.5ms at concurrency 8) is only 35% higher than single-request latency, indicating efficient batch formation.

### Prometheus Metrics

Triton's built-in metrics endpoint confirmed production readiness: 775 successful inferences with zero failures, 55 batch executions (average batch size of 14.1 requests — dynamic batching working as configured), and 9.4 seconds total compute time versus 835.8 seconds total queue time. The queue dominance confirms that the GPU processes requests far faster than they arrive at sub-saturation concurrency, with requests spending most of their time waiting to be batched.

GPU memory footprint was minimal: 345 MB for the embedding model, leaving the vast majority of the GPU available for concurrent model serving.

### Key Findings

Triton's architecture is fundamentally different from vLLM's. Where vLLM is purpose-built for autoregressive LLM generation with continuous batching and KV cache management, Triton is a general-purpose model server supporting any ONNX, TensorRT, or PyTorch model with configurable batching strategies. For non-generative models like embeddings, Triton is the natural deployment choice.

The batching efficiency ratio (14.1 average batch size from preferred targets of [8, 16, 32]) shows that Triton's scheduler effectively aggregates concurrent requests while respecting the 100ms max queue delay. At low concurrency, Triton serves requests individually with sub-200ms latency; at high concurrency, it batches aggressively for 3.5x throughput gain.

The 345 MB memory footprint opens the door to multi-model deployment — running the embedding model alongside an LLM on separate GPUs, or even co-locating multiple small models on a single GPU.

---

## Challenges & Resolutions

### 1. Tensor Parallelism on PCIe-Limited Hardware

**Challenge:** Week 3 established that GPUs 1-3 are limited to PCIe 3.0 x1 bandwidth (1.09 GB/s), making tensor parallelism unviable. However, GPU 0 (PCIe 4.0 x16) and GPU 1 communicate through the PCIe root complex, which provides better bandwidth than the x1 slots.

**Resolution:** vLLM's TP=2 deployment across GPUs 0-1 worked, but the all-reduce synchronization overhead at each of the 48 layers creates measurable per-token latency. The resulting 38.6 tok/s single-request rate (vs 53.1 for 7B on one GPU) reflects both the larger model compute requirement and the inter-GPU communication cost. This confirms the Week 3 finding that tensor parallelism should only be used when the model genuinely exceeds single-GPU memory capacity.

### 2. Triton Model Repository Setup

**Challenge:** Triton requires a specific directory layout, ONNX model export with correct tensor specifications, and a `config.pbtxt` that precisely matches the model's input/output signature.

**Resolution:** Exported the embedding model to ONNX with fixed sequence length (128 tokens), configured Triton with dynamic batching, and validated end-to-end through the gRPC client API. The setup process, while more manual than vLLM's single-command deployment, provides full control over batching behavior and monitoring.

### 3. KV Cache Memory Pressure at 14B Scale

**Challenge:** With 14B model weights distributed across 2 GPUs, significantly less VRAM remained for KV cache — only 5.81 GiB versus 13.97 GiB for the 3B model on one GPU. Combined with 1.71x higher per-token KV cache cost, total token capacity dropped from ~130K to 63,440.

**Resolution:** This is a fundamental capacity constraint, not a bug. At 4,096 tokens per request, the 14B model can handle approximately 15 concurrent requests before exhausting KV cache — far fewer than the 3B model's ~32 concurrent long-context requests. Production deployments of 14B models require either shorter context lengths, KV cache quantization, or additional GPUs.

---

## Technical Concepts Explored

### Tensor Parallelism Communication

With TP=2, each forward pass through a transformer layer requires an all-reduce operation to synchronize partial results between GPUs. For Qwen 2.5 14B with 48 layers, that's 48 synchronization points per token generated. The communication uses NCCL (NVIDIA Collective Communications Library), which coordinates GPU-to-GPU data movement through the optimal available path — in this case, the PCIe root complex since direct P2P (peer-to-peer) access is not available between GPUs 0 and 1 on this motherboard.

### Qwen 2.5 14B Architecture

Qwen 2.5 14B uses Grouped Query Attention with 40 query heads sharing 8 KV heads (5:1 ratio), 128 dimensions per head, and a total hidden dimension of 5,120 across 48 transformer layers. The 5:1 GQA ratio is more aggressive than Llama's 3:1, further reducing KV cache requirements relative to model size while maintaining quality through the larger number of independent query projections.

### Triton Dynamic Batching

Triton's dynamic batcher queues incoming requests and forms batches based on configurable parameters: preferred batch sizes and maximum queue delay. When requests arrive at a rate that fills preferred batch sizes within the delay window, throughput scales efficiently. The scheduler makes a continuous tradeoff between latency (waiting to form larger batches) and throughput (processing more requests per GPU kernel launch). The 100ms max delay setting means no individual request waits more than 100ms for batch formation, even at low load.

---

## Files Created

```
phase-2-production/week-06/
├── exp1_qwen14b_tp2_benchmark.py
├── exp2_7b_vs_14b_economics.py
├── exp3_triton_embedding_test.py
└── results/
    ├── qwen14b_tp2_benchmark.txt
    ├── 7b_vs_14b_economics.txt
    └── triton_embedding_test.txt
```

---

## Cumulative Progress

| Week | Key Achievement | Throughput Milestone |
|------|----------------|---------------------|
| 1 | Llama 3.2 3B baseline, memory model | 84 tok/s (single), ~5,000 tok/s (peak batch) |
| 2 | TensorRT pipeline, ONNX limitations | Validated that generic pipelines fail for LLMs |
| 3 | 4-GPU topology, parallelism strategies | 7,422 tok/s (4-GPU data parallel), PCIe x1 discovery |
| 4 | vLLM single-GPU, PagedAttention | 106 tok/s (single), 4,731 tok/s (peak) |
| 5 | vLLM multi-GPU, sustained load | 18,053 tok/s (4-GPU), 7.12x over transformers |
| **6** | **14B scaling, Triton deployment** | **316.5 tok/s (14B), 23.2 req/s (Triton embedding)** |

---

## Next Steps: Week 7 Preview

### Triton Inference Server Deep Dive

With the embedding model deployment validated, Week 7 will expand Triton to multi-model serving: deploying both the embedding model and a generation model simultaneously, configuring model-specific GPU assignments, and measuring the operational overhead of managing multiple models through a single Triton instance.

### Dynamic Batching Optimization

The current embedding model batching configuration used default-reasonable settings. Week 7 will systematically tune batch sizes, queue delays, and instance counts to find optimal configurations for different latency/throughput tradeoff points.

### Triton vs vLLM Comparison

With both frameworks now operational, a direct comparison for LLM serving becomes possible. Triton's vLLM backend integration will be tested against standalone vLLM to understand what Triton adds (or costs) as an orchestration layer.

---

## Conclusion

Week 6 delivered two critical insights for production AI infrastructure. First, model size scaling has severely diminishing returns on consumer GPU hardware: doubling parameters from 7B to 14B requires 2x the GPUs, produces only 0.73x the single-request throughput, and costs 19.69x more per token at peak load. The non-linear throughput degradation under concurrency — from a 1.38x gap at single request to 9.84x at concurrency 256 — means that larger models don't just cost more per token, they cost *increasingly* more per token as utilization rises. This shapes how production systems should be architected: use the smallest model that meets quality requirements, and reserve larger models for quality-critical paths where the cost premium is justified.

Second, Triton Inference Server brings production model serving beyond LLMs. The embedding model deployment demonstrated Triton's core value proposition: a unified serving infrastructure supporting dynamic batching, Prometheus monitoring, and multi-model orchestration with minimal GPU memory overhead. Combined with vLLM for LLM-specific workloads, the two frameworks cover the full spectrum of production inference needs.

**Week 6 Status:** ✅ Complete — All objectives met

---

## Appendix: Q&A Topics Covered

### KV Cache Calculation for Qwen 2.5 14B

The per-token KV cache size calculation for Qwen 2.5 14B:

```
2 (K and V) × 48 layers × 8 KV heads × 128 dim × 2 bytes (FP16) = 196,608 bytes ≈ 192 KB/token
```

The "128-dim per head" refers to the dimensionality of each attention head's key and value vectors. With 5,120 total hidden dimensions and 40 query heads, each head operates on 128 dimensions (5,120 / 40 = 128). The 8 KV heads use the same 128-dimensional space, with groups of 5 query heads sharing each KV head.

### Tensor Parallelism (TP=2) Mechanics

TP=2 splits model weights across 2 GPUs such that each GPU holds half of every weight matrix. For each token, both GPUs compute partial results simultaneously, then synchronize via all-reduce. The communication pattern is: each GPU computes its portion of the attention and FFN layers → all-reduce to combine partial sums → proceed to next layer. With 48 layers, this creates 48 synchronization barriers per token.

### P2P and NCCL

P2P (peer-to-peer) refers to direct GPU-to-GPU memory access without involving the CPU or system memory. On hardware with NVLink, P2P provides high-bandwidth direct paths between GPUs. On this system, P2P is unavailable between most GPU pairs due to the PCIe topology, so GPU communication routes through system memory via the PCIe root complex.

NCCL (NVIDIA Collective Communications Library) abstracts multi-GPU communication primitives — all-reduce, all-gather, broadcast — and automatically selects the optimal transport based on available hardware paths. When P2P/NVLink is available, NCCL uses it; otherwise, it falls back to shared memory or PCIe transfers through the host.

---

*Report generated: February 2026*  
*Hardware: 4x RTX 3090, Gigabyte B650 Eagle AX, Ubuntu 24.04, CUDA 12.6*  
*Frameworks: vLLM 0.13.0 (V1 engine), NVIDIA Triton Inference Server, ONNX Runtime*  
*Models: Qwen 2.5 14B Instruct (TP=2), Mistral 7B Instruct v0.3, all-MiniLM-L6-v2*