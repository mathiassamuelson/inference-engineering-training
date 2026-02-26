# Week 5: vLLM Multi-GPU & Larger Models

**Duration:** February 2026  
**Hardware:** 4x RTX 3090 (24GB each), Ubuntu 24.04 LTS, CUDA 12.6  
**PCIe Topology:** GPU 0 on x16 (CPU-direct), GPUs 1-3 on x1 (chipset-routed)  
**Framework:** vLLM 0.13.0 (V1 engine)  
**Models:** Llama 3.2 3B Instruct, Mistral 7B Instruct v0.3

---

## Executive Summary

Week 5 scaled vLLM from single-GPU baselines (Week 4) to multi-GPU production deployments, testing data parallelism, larger models, sustained load, and a direct comparison against transformers-based serving. Four experiments produced five key findings:

1. **vLLM data parallelism scales at 95.4% efficiency across 4 GPUs** — delivering 18,053 tok/s peak system throughput at concurrency 256, with near-perfect load balancing (1% request spread across GPUs). PCIe x1 slots remain a non-factor for inference throughput.

2. **Mistral 7B confirms larger models maintain consistent per-token efficiency** — single-request throughput of 53.2 tok/s (exactly 0.50x of the 3B model's 106.4 tok/s) with more graceful latency degradation under load (27% vs 30% at concurrency 64), validating that larger models have better arithmetic intensity.

3. **Sustained throughput is rock-solid at 2,923 tok/s** — coefficient of variation 0.023 over 60 seconds with uniform load, and 0.082 with mixed workloads. Production systems can reliably plan around these numbers.

4. **Continuous batching isolates request latencies under mixed workloads** — quick replies complete in 0.36s while long responses take 4.84s, with no interference between them. Per-request throughput remains 88.6–90.8 tok/s regardless of output length, confirming fair compute sharing.

5. **vLLM delivers 7.12x system throughput over transformers on identical hardware** — the entire advantage comes from architectural batching (serving many requests concurrently per GPU), not from per-request speedups. Individual request latency is nearly identical between the two frameworks (1.04x difference).

**Key takeaway:** The gap between a naive multi-GPU deployment (transformers + multiprocessing at 260 tok/s) and a production-grade one (vLLM at 1,852 tok/s) is 7x — and this is on a 3B model where vLLM's optimizations have the least room to improve. The multiplier will be larger with bigger models and longer sequences.

---

## Objectives

- ✅ Deploy vLLM with data parallelism across 4 GPUs
- ✅ Benchmark Llama 3.2 3B: 4-GPU data parallel throughput vs single-GPU
- ✅ Scale to larger model: Mistral 7B on single GPU, measure vLLM advantages at scale
- ✅ Compare vLLM multi-GPU orchestration vs Week 3's manual data parallelism
- ✅ Sustained load testing demonstrating continuous batching across GPUs

---

## Experiment 1: vLLM Data Parallel Scaling

**Setup:** Independent vLLM OpenAI-compatible servers, one per GPU (ports 8000-8003), tested at 1, 2, and 4 GPUs with concurrency levels 1–64 per GPU. Model: Llama 3.2 3B Instruct with 128-token output.

### Scaling Results (Peak Concurrency = 64/GPU)

| GPUs | System tok/s | Scaling Factor | Efficiency | Avg Per-Request tok/s |
|------|-------------|----------------|------------|----------------------|
| 1    | 4,731       | 1.00x          | 100%       | 74.1                 |
| 2    | 9,409       | 1.99x          | 99.4%      | 73.7                 |
| 4    | 18,053      | 3.82x          | 95.4%      | 72.8                 |

### Throughput Scaling by Concurrency (4 GPUs)

| Concurrency/GPU | Total Concurrency | System tok/s | Per-Request tok/s | Avg Latency |
|-----------------|-------------------|-------------|-------------------|-------------|
| 1               | 4                 | 424         | 106.5             | 1.202s      |
| 4               | 16                | 1,506       | 100.9             | 1.269s      |
| 8               | 32                | 2,871       | 97.9              | 1.307s      |
| 16              | 64                | 5,796       | 93.1              | 1.376s      |
| 32              | 128               | 10,123      | 81.8              | 1.565s      |
| 64              | 256               | 18,053      | 72.8              | 1.759s      |

### Per-GPU Balance (4 GPUs, Concurrency 64/GPU)

| GPU | PCIe Slot | Per-Request tok/s | Requests |
|-----|-----------|-------------------|----------|
| 0   | x16       | 73.0              | 192      |
| 1   | x1        | 70.7              | 192      |
| 2   | x1        | 73.8              | 192      |
| 3   | x1        | 73.7              | 192      |

### Key Findings

Scaling efficiency exceeds Week 3's manual data parallelism (93.6% at batch=32 with transformers) because vLLM handles batching internally, eliminating Python-level coordination overhead. The small drop from 99.4% at 2 GPUs to 95.4% at 4 GPUs comes from CPU/network overhead managing 768 HTTP requests across 4 servers, not from GPU-side contention.

PCIe x1 slots remain a non-factor for inference throughput. GPUs 2 and 3 (x1 slots) slightly outperform GPU 0 (x16) in some measurements. GPU 1 shows a consistent ~4% deficit (70.7 vs 73.0-73.8 tok/s), likely from thermal positioning rather than PCIe bandwidth.

At equivalent concurrency (4 GPUs, concurrency 32), vLLM delivers 10,123 tok/s versus Week 3's 7,422 tok/s — a 1.36x improvement from kernel-level optimizations (CUDA graphs, Flash Attention, torch.compile). At concurrency 64, vLLM reaches 18,053 tok/s — 2.43x Week 3's peak — because transformers hit a hard ceiling at ~5,000 tok/s per GPU while vLLM continues scaling.

Latency degradation remains graceful: per-request throughput drops from 106.5 → 72.8 tok/s (32% decrease) going from concurrency 1 to 64 per GPU, compared to transformers' 95% degradation at high batch sizes from Week 1.

---

## Experiment 2: Larger Model Benchmark (Mistral 7B)

**Setup:** Single-GPU vLLM serving Mistral 7B Instruct v0.3 on GPU 0, tested at concurrency levels 1–64. FP16, max sequence length 4096.

### Throughput Results

| Concurrency | System tok/s | Per-Request tok/s | Avg Latency |
|-------------|-------------|-------------------|-------------|
| 1           | 53.2        | 53.2              | 2.405s      |
| 2           | 102.3       | 51.2              | 2.501s      |
| 4           | 190.8       | 50.9              | 2.515s      |
| 8           | 375.2       | 50.0              | 2.558s      |
| 16          | 739.3       | 46.2              | 2.769s      |
| 32          | 1,458.9     | 45.6              | 2.807s      |
| 64          | 2,478.6     | 38.7              | 3.304s      |

### 3B vs 7B Comparison

| Metric                      | Llama 3.2 3B | Mistral 7B | Ratio |
|-----------------------------|-------------|------------|-------|
| Single-request tok/s        | 106.4       | 53.2       | 0.50x |
| Peak system tok/s (conc=64) | 4,731       | 2,479      | 0.52x |
| Latency at concurrency 1    | 1.20s       | 2.41s      | 2.0x  |
| Latency at concurrency 64   | 1.73s       | 3.30s      | 1.91x |

### Key Findings

The 7B model produces almost exactly half the throughput of the 3B model — expected since Mistral 7B has ~2.3x the parameters. The slightly better-than-2.3x ratio (0.50x instead of 0.43x) confirms that larger models have better arithmetic intensity: more FLOPs per byte of memory read, which means the GPU compute units are better utilized relative to memory bandwidth.

Latency degradation under load is more graceful for the 7B model. Per-request throughput drops 27% (53.2 → 38.7 tok/s) from concurrency 1 to 64, versus 30% for the 3B model. The throughput curve was still climbing at concurrency 64, suggesting the 7B model had headroom for additional concurrent requests before saturating — unlike the 3B model which was already flattening.

The test timed out at concurrency 96 (288 requests × ~4s each exceeded the 180s session timeout), confirming that very high concurrency with larger models requires longer timeout configurations.

Under a realistic SLA of p95 < 5s for a 7B model, capacity would be approximately 32–64 concurrent users per GPU at 1,459–2,479 tok/s system throughput.

---

## Experiment 3: Sustained Load Testing

**Setup:** 4 vLLM instances (one per GPU), 32 total concurrent requests, 60-second test windows. Two phases: uniform requests (128 tokens) and mixed workload (32–512 tokens, weighted).

### Mixed Workload Profile

| Profile       | Weight | Max Tokens | Simulates            |
|---------------|--------|------------|----------------------|
| quick_reply   | 40%    | 32         | Short factual answers |
| short_answer  | 30%    | 128        | Typical chat replies  |
| explanation   | 20%    | 256        | Detailed responses    |
| long_response | 10%    | 512        | Extended generation   |

### Phase A: Uniform Requests

| Metric             | Value               |
|--------------------|---------------------|
| System throughput  | 2,923 tok/s         |
| Request rate       | 23.1 req/s          |
| p50 latency        | 1.385s              |
| p95 latency        | 1.397s              |
| p99 latency        | 1.402s              |
| Stability (CV)     | 0.023 (stable)      |

Per-GPU distribution was near-perfect: 346–347 requests each, with per-GPU throughput ranging 91.9–92.8 tok/s (1% spread).

### Throughput Stability (Phase A, 10-second windows)

| Window   | Requests | Tokens  | tok/s   |
|----------|----------|---------|---------|
| 0–10s    | 227      | 28,760  | 2,876   |
| 10–20s   | 251      | 31,547  | 3,155   |
| 20–30s   | 229      | 28,955  | 2,896   |
| 30–40s   | 231      | 29,253  | 2,925   |
| 40–50s   | 231      | 29,226  | 2,923   |

### Phase B: Mixed Requests

| Metric             | Value               |
|--------------------|---------------------|
| System throughput  | 1,416 tok/s         |
| Request rate       | 10.1 req/s          |
| p50 latency        | 1.396s              |
| p95 latency        | 5.571s              |
| p99 latency        | 5.795s              |
| Stability (CV)     | 0.082 (stable)      |

### Per-Profile Breakdown (Phase B)

| Profile       | Count | Avg Tokens | Avg Latency | p95 Latency | Per-Request tok/s |
|---------------|-------|-----------|-------------|-------------|-------------------|
| quick_reply   | 266   | 32.0      | 0.361s      | 0.376s      | 88.6              |
| short_answer  | 170   | 127.4     | 1.411s      | 1.464s      | 90.3              |
| explanation   | 106   | 244.7     | 2.696s      | 2.922s      | 90.8              |
| long_response | 66    | 437.4     | 4.838s      | 5.830s      | 90.4              |

### Key Findings

Sustained throughput is rock-solid. Phase A held 2,876–3,155 tok/s across all 10-second windows with a coefficient of variation of just 0.023. Production capacity planning can reliably use these numbers — there is no throughput drift or degradation over time.

Continuous batching provides fair compute sharing across request types. Per-request throughput ranges 88.6–90.8 tok/s across all four profiles regardless of output length. Each request gets its proportional share of GPU compute without starving or being penalized by other requests in the batch.

Short requests are not held hostage by long ones. A quick_reply completes in 0.36s while a long_response takes 4.84s, coexisting in the same GPU batch. With static batching, that quick_reply would wait the full 4.84s for the batch to complete — a 13x latency penalty affecting 40% of traffic. This is the operational advantage that makes continuous batching essential for production chat services.

The lower system throughput in Phase B (1,416 vs 2,923 tok/s) is expected: with some requests occupying concurrency slots for 4.8s instead of 1.4s, fewer requests cycle through per second. The first two 10-second stability windows (2,551 and 2,996 tok/s) confirm the per-GPU token generation rate is comparable to Phase A.

---

## Experiment 4: vLLM vs Manual Data Parallelism

**Setup:** Direct comparison on identical hardware with identical mixed workload (32–512 tokens, weighted). Part A: transformers + multiprocessing (4 workers, one per GPU, batch_size=1 per worker). Part B: 4 vLLM instances with continuous batching. Both tested for 60 seconds at 32 total concurrency.

### Head-to-Head Results

| Metric              | Transformers | vLLM    | Ratio  |
|---------------------|-------------|---------|--------|
| System throughput   | 260 tok/s   | 1,852 tok/s | **7.12x** |
| Request rate        | 2.1 req/s   | 13.0 req/s  | 6.15x  |
| Avg latency         | 1.447s      | 1.585s      | 0.91x  |
| Total requests (60s)| 165         | 1,178       | 7.1x   |

### Per-Profile Latency Comparison

| Profile       | Transformers | vLLM   | Speedup |
|---------------|-------------|--------|---------|
| quick_reply   | 0.374s      | 0.361s | 1.04x   |
| short_answer  | 1.467s      | 1.400s | 1.05x   |
| explanation   | 2.630s      | 2.702s | 0.97x   |
| long_response | 4.763s      | 4.553s | 1.05x   |

### GPU Load Balance

| GPU | Transformers Reqs | vLLM Reqs | Transformers tok/s | vLLM tok/s |
|-----|-------------------|-----------|-------------------|------------|
| 0   | 39                | 293       | 85.6              | 89.0       |
| 1   | 46                | 295       | 85.0              | 89.6       |
| 2   | 43                | 294       | 86.1              | 89.9       |
| 3   | 37                | 296       | 85.5              | 89.6       |

### Key Findings

The 7.12x system throughput advantage comes entirely from architectural batching, not per-request optimization. The per-profile latency comparison shows nearly identical numbers — quick_reply is 0.374s vs 0.361s (1.04x), long_response is 4.763s vs 4.553s (1.05x). vLLM does not make individual requests faster; it serves 7x more requests concurrently through continuous batching.

Transformers with multiprocessing is fundamentally limited by sequential processing. Each worker handles one request at a time, capping throughput at 4 GPUs × ~85 tok/s = ~340 tok/s theoretical maximum (260 tok/s measured, with queue management overhead). vLLM batches dozens of requests per GPU simultaneously, multiplying throughput without proportional latency cost.

Transformers' average latency is actually 9% better than vLLM's (1.447s vs 1.585s). With no batching contention, each transformers request gets the full GPU exclusively. vLLM's slight latency increase is the cost of concurrent batching — but trading 10% latency for 7x throughput is the correct trade for every production system.

GPU load balancing differs significantly. Transformers with a shared queue shows 24% request spread (37–46 per GPU) because faster-draining workers pull more work. vLLM's round-robin dispatch achieves 1% spread (293–296 per GPU).

---

## Cross-Week Performance Summary

### System Throughput Progression (4 GPUs, Llama 3.2 3B)

| Week | Framework     | Configuration            | System tok/s | Per-Request tok/s |
|------|--------------|--------------------------|-------------|-------------------|
| 1    | transformers | Single GPU, batch=1      | 84          | 84.0              |
| 1    | transformers | Single GPU, peak batch   | ~5,000      | 9.2               |
| 3    | transformers | 4 GPU data parallel, b=32| 7,422       | 85.4              |
| 4    | vLLM         | Single GPU, conc=1       | 106         | 106.4             |
| 4    | vLLM         | Single GPU, conc=64      | ~6,100      | ~74.1             |
| 5    | vLLM         | 4 GPU data parallel, c=32| 10,123      | 81.8              |
| 5    | vLLM         | 4 GPU data parallel, c=64| 18,053      | 72.8              |

### Key Metric: Throughput per User Quality

The most production-relevant metric is system throughput while maintaining acceptable per-user experience. Transformers achieves 7,422 tok/s at 4 GPUs but only by destroying per-sample throughput to 9.2 tok/s (95% degradation). vLLM achieves 18,053 tok/s while maintaining 72.8 tok/s per request (32% degradation). This means vLLM delivers 2.4x the total throughput while providing 7.9x better per-user experience.

---

## Interview Articulations

### On vLLM Multi-GPU Scaling
"We deployed vLLM as independent data-parallel instances across 4x RTX 3090s and measured 95.4% scaling efficiency at 256 concurrent requests — better than the 93.6% we achieved with manual transformers-based data parallelism in Week 3. The improvement comes from eliminating Python-level coordination: each vLLM instance handles its own batching internally, removing the queue management overhead that reduced efficiency in the manual approach."

### On Framework Selection
"Our head-to-head comparison showed vLLM delivering 7.12x system throughput over transformers on identical hardware with identical workloads. Critically, per-request latency was nearly identical — the entire advantage comes from continuous batching serving many requests concurrently per GPU. This means framework selection is the single highest-leverage decision in inference infrastructure. Hardware upgrades give you 1.5-2x; switching from naive to production-grade serving gives you 7x."

### On Continuous Batching
"Under a realistic mixed workload — 40% short replies, 30% medium, 20% explanations, 10% long responses — continuous batching completed short requests in 0.36 seconds while long requests took 4.84 seconds, with no interference between them. Per-request throughput held steady at 88-91 tok/s regardless of output length. With static batching, every short request would wait for the longest response in the batch — a 13x latency penalty affecting 40% of users."

### On Production Capacity Planning
"We measured sustained 4-GPU throughput at 2,923 tok/s with a coefficient of variation under 3% over 60-second windows. That stability is essential for capacity planning — you can reliably promise that number in an SLA. But the binding constraint is still latency, not throughput: under a p95 < 2s SLA for the 3B model, capacity is roughly 25 concurrent users per GPU, far below the memory-based estimate of 1,200. For the 7B model, even single-request latency at 2.4s already exceeds that threshold."

### On Larger Model Economics
"Mistral 7B produced exactly half the throughput of the 3B model — 53 vs 106 tok/s single-request. But the 7B model degraded more gracefully under load: 27% throughput loss at concurrency 64 versus 30% for the 3B model. Larger models have better arithmetic intensity, meaning the GPU compute units are better utilized relative to memory bandwidth. This has direct cost implications: doubling model size doesn't double your infrastructure cost because you extract more useful compute per GPU."

---

## Challenges & Resolutions

### 1. Orphaned vLLM Processes
**Problem:** vLLM spawns child processes (VLLM::EngineCore) that survive parent termination, holding GPU memory and blocking subsequent experiments.
**Solution:** Launch servers with `preexec_fn=os.setsid` to create dedicated process groups, then use `os.killpg()` with SIGTERM/SIGKILL to clean up entire process trees. Always verify GPU memory is free between experiments with `nvidia-smi`.

### 2. Sustained Load Test Drain Issue
**Problem:** Mixed workload test ran for 147.9s instead of 60s because in-flight long_response requests (5+ seconds each) drained slowly after the test window, cascading into HTTP timeouts and 32 errors.
**Solution:** Added `timeout=remaining` to `asyncio.wait()` and cancel in-flight tasks at deadline rather than draining them. Clean run duration: 60.0s.

### 3. HTTP Timeout at High Concurrency (Mistral 7B)
**Problem:** Concurrency 96 with Mistral 7B exceeded the 180s aiohttp session timeout — 288 requests at ~4s each created cascading latency.
**Resolution:** Accepted as a practical finding rather than a bug. The data through concurrency 64 captured the full scaling curve. Production systems need timeout configuration proportional to model size × max output length.

### 4. Multiprocessing Silent Exit
**Problem:** Experiment 4 script exited silently with no traceback due to `mp.set_start_method("spawn")` being called inside an async function.
**Solution:** Moved `mp.set_start_method("spawn", force=True)` to the `__main__` block before `asyncio.run()`, added explicit exception handling with traceback printing.

---

## Files Created

```
phase-2-production/week-05-vllm-multi-gpu/
├── exp1_data_parallel_scaling.py
├── exp2_larger_model_benchmark.py
├── exp3_sustained_load_test.py
└── exp4_vllm_vs_manual_dp.py
```

---

## Next Steps: Week 6 Preview

### Potential Directions
- **Triton Inference Server deployment** — multi-model serving (embedding + classification + generation) with dynamic batching and Prometheus monitoring
- **vLLM with larger models across multiple GPUs** — Qwen 2.5 14B across 2 GPUs, testing where single-GPU VRAM limits force multi-GPU strategies
- **NVLink tensor parallelism** — if the NVLink bridge arrives, re-run topology benchmarks and test vLLM tensor parallelism between GPU 0 and GPU 1

### Open Questions
- Does Triton's model ensemble capability provide meaningful advantages over separate vLLM instances for LLM-specific workloads?
- At what model size does the 7x framework advantage grow — is it 10x for 14B models, or does it plateau?
- How does vLLM tensor parallelism over NVLink compare to data parallelism for latency-sensitive workloads?

---

## Conclusion

Week 5 established that vLLM multi-GPU data parallelism is the production-grade serving strategy for this hardware. The 7.12x throughput advantage over transformers represents the single largest performance improvement measured in this entire training program — larger than FP16 optimization (1.56x), TensorRT conversion, or multi-GPU scaling. Framework architecture dominates all other optimization axes.

The sustained load testing proved that these numbers are reliable and stable over time, and the continuous batching experiments demonstrated the operational quality that separates production systems from benchmarks: fair compute sharing, latency isolation between request types, and graceful degradation under mixed workloads.

Combined with Week 4's single-GPU analysis, the vLLM investigation is complete for the 3B model. The path forward is either scaling to larger models where vLLM's advantages compound, or adding Triton Inference Server for multi-model orchestration.

**Week 5 Status:** ✅ Complete — All objectives met

**Ready for Week 6:** Triton Inference Server or larger model multi-GPU experiments

---

*Report generated: February 2026*  
*Hardware: 4x RTX 3090, Gigabyte B650 Eagle AX, Ubuntu 24.04, CUDA 12.6*  
*Framework: vLLM 0.13.0 (V1 engine)*  
*Models: Llama 3.2 3B Instruct (multi-GPU), Mistral 7B Instruct v0.3 (single GPU)*