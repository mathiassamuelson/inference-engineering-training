# Training Plan: NVIDIA Stack → AI Engineering/Product Management (Revised)

*Last updated: March 10, 2026 — reflects actual progress through Week 7*

---

## Phase 1: Foundation & Baselines (Weeks 1-4) ✅ Complete

*Goal: Establish baseline measurements, understand hardware constraints, and validate the need for production inference frameworks*

### Week 1: Inference Baselines & Capacity Analysis ✅

- Benchmarked Llama 3.2 3B on transformers library (FP32, FP16, multi-GPU)
- Batch size scaling sweep (1 → 1200)
- Memory model with linear regression (R² = 0.9999)
- **Key findings:** FP16 gives 1.56x speedup (memory-bandwidth limited, not compute). Throughput plateaus at ~5,000 tok/s total regardless of batch size — 95% per-sample degradation. Theoretical capacity (1,200 users) vs practical capacity (100-150 users) gap. Small models don't benefit from multi-GPU with transformers
- **Hardware:** 2x RTX 3090

### Week 2: TensorRT Optimization Pipeline ✅

- Established PyTorch → ONNX → TensorRT conversion pipeline
- Benchmarked SimpleNet (1M params): 1.17x speedup (overhead-dominated at 30μs inference)
- Attempted Llama 3.2 3B: ONNX export OOM (>60GB RAM needed for tied weight duplication)
- Llama 3.2 1B via Optimum: TensorRT *slower* than PyTorch (81 vs 183 tok/s) due to CPU device placement
- Evaluated TensorRT-LLM: blocked by CUDA 13 requirement
- **Key finding:** Generic ONNX/TensorRT pipelines don't work for LLMs. Production optimization requires purpose-built frameworks that manage device placement, KV cache, and memory at the framework level
- **Hardware:** 4x RTX 3090 (GPUs 3-4 installed this week)

### Week 3: Multi-GPU Orchestration & Hardware Topology ✅

- Discovered critical PCIe topology: GPU 0 on PCIe 4.0 x16 (~25 GB/s), GPUs 1-3 on PCIe 3.0 x1 (~1 GB/s)
- Data parallelism: 93.6% scaling efficiency at 4 GPUs (batch=32), 7,422 tok/s total
- Pipeline parallelism: 8-18% throughput loss from synchronization overhead
- CUDA streams: 99.8% cross-GPU concurrent efficiency; multi-stream same-GPU only 11.6% improvement
- Ring all-reduce: 378.9ms for 32MB — tensor parallelism completely unviable on PCIe x1
- **Key finding:** Data parallelism is the only viable multi-GPU strategy on this hardware without NVLink. PCIe x1 has zero impact on inference once models are GPU-resident
- **Hardware:** 4x RTX 3090

### Week 4: vLLM Single-GPU Fundamentals ✅

- Installed vLLM 0.13.0 (V1 engine), verified PagedAttention, CUDA graphs, fused kernels at startup
- Single-GPU throughput: ~1.3x improvement over transformers baseline (modest for small model)
- Concurrent user simulation demonstrating continuous batching and graceful degradation
- PagedAttention memory analysis: 112 KB/token (vs 260 KB measured in Week 1, vs 344 KB incorrect MHA calculation)
- Corrected Week 1 KV cache calculation: Llama 3.2 3B uses GQA with 8 KV heads (not 24), giving 3:1 ratio
- SLA-driven capacity planning: memory says 1,200 users, throughput says 100-150, SLA (p95 < 2s) says ~25
- **Key finding:** vLLM's value for small models is primarily operational (request queuing, graceful degradation, memory efficiency) rather than raw throughput. The transformative capabilities compound with model size, context length, and deployment complexity

---

## Phase 2: Production Inference at Scale (Weeks 5-8)

*Goal: Master multi-GPU production serving, larger models, and multi-model orchestration*

### Week 5: vLLM Multi-GPU & Sustained Load Testing ✅

- Deployed vLLM with data parallelism across 4 GPUs (separate vLLM instances per GPU)
- Measured **7.12x throughput advantage** over transformers — the largest single performance improvement in the training program, purely from framework architecture
- Sustained load testing confirmed throughput stability over extended runs with no memory fragmentation or degradation
- Mixed workload continuous batching: fair compute sharing between short and long requests, latency isolation, graceful degradation under overload
- Benchmarked Mistral 7B Instruct v0.3 on single GPU, confirming vLLM advantages scale with model size
- Production metrics: p50/p95/p99 latency under sustained concurrent load
- **Key finding:** Framework architecture dominates all other optimization axes — more impactful than FP16 optimization (1.56x), TensorRT conversion, or multi-GPU scaling
- **Models tested:** Llama 3.2 3B Instruct (multi-GPU), Mistral 7B Instruct v0.3 (single GPU)

### Week 6: Larger Model Scaling & Triton Introduction ✅

- Scaled to Qwen 2.5 14B with TP=2 across GPU0+GPU1 (PCIe): 316.5 tok/s peak — severe throughput cliff vs 7B
- 7B vs 14B economics: 19.69x cost-per-token penalty for 14B over 7B at peak load (later revised — see Week 7)
- Deployed embedding model on Triton Inference Server with dynamic batching: 23.2 req/s peak, 3.5x batching speedup
- Prometheus metrics integration for Triton
- **Key finding (revised by Week 7):** The 14B throughput collapse was caused by PCIe x1 all-reduce overhead, not model size per se. The finding that "14B has steep cost curves" was specific to PCIe-connected tensor parallelism
- **Models tested:** Qwen 2.5 14B Instruct (TP=2 PCIe), sentence-transformers/all-MiniLM-L6-v2 (Triton)

### Week 7: NVLink Tensor Parallelism ✅

- Installed AORUS NVLink bridge; topology verified: GPU0 ↔ GPU2 (NV4, ~100 GB/s bidirectional)
- Reran Qwen 2.5 14B TP=2 on NVLink pair (GPU0+GPU2): **3,018 tok/s peak** vs 316.5 tok/s on PCIe — **9.53x improvement**
- Concurrency sweep confirmed compute saturation at concurrency 128 (correct bottleneck)
- Latency distribution (30 trials, concurrency 128): mean 2.364s, stdev 0.015s, p50→p99 spread 35ms (0.6% CV)
- Output length sweep: throughput rises 2,279 → 3,018 tok/s from 25→200 output tokens (prefill amortization)
- **Key finding:** NVLink transforms tensor parallelism from communication-bottlenecked to compute-saturated. The Week 6 PCIe results were measuring interconnect overhead, not model capability. Qwen 2.5 14B on NVLink TP=2 is a viable production serving configuration
- **Models tested:** Qwen 2.5 14B Instruct (TP=2 NVLink, GPU0+GPU2)

### Week 8: Triton Inference Server Deep Dive & Framework Comparison

- **Triton multi-model serving:** deploy embedding model + generation model simultaneously across GPUs, measure operational overhead
- **Dynamic batching optimization:** systematic tuning of batch sizes, queue delays, and instance counts
- **Prometheus metrics integration:** production observability for multi-model Triton deployment
- **NVLink TP=2 vs 4-GPU data parallel comparison:** NVLink 14B (2 GPUs) vs data parallel 3B (4 GPUs) — direct quality vs. concurrency tradeoff measurement
- **Framework decision matrix:** when to use Triton vs vLLM vs both (Triton + vLLM backend)
- **Deliverable:** Multi-model inference API with performance monitoring + framework selection guide with measured data

---

## Phase 3: Optimization & Quantization (Weeks 9-12)

*Goal: Master model compression and optimization techniques*

### Week 9: Quantization Methods (AWQ, GPTQ)

- Apply AWQ and GPTQ quantization to Llama 3.2 3B and a 7-14B model
- Measure quality degradation: perplexity, generation coherence, task accuracy
- Benchmark INT4/INT8 vs FP16: throughput, latency, memory savings
- **Deliverable:** Quality vs performance tradeoff analysis across quantization methods

### Week 10: Quantized Model Serving with vLLM

- Serve quantized models through vLLM: measure end-to-end production impact
- Capacity planning with quantization: how many more concurrent users per GPU?
- **Deliverable:** Quantization deployment guide with measured quality/performance tradeoffs

### Week 11: Speculative Decoding & KV Cache Compression

- Implement speculative decoding with a draft model (Llama 3.2 1B → 3B)
- Measure acceptance rate, latency improvement, and throughput impact
- KV cache compression techniques: sliding window, prefix caching
- **Deliverable:** Speculative decoding analysis with acceptance rate breakdown

### Week 12: NSight Profiling & Bottleneck Analysis

- Profile inference workloads with NVIDIA NSight Systems and NSight Compute
- Identify compute vs memory bandwidth bottlenecks at the kernel level
- Profile vLLM's PagedAttention and continuous batching kernel execution
- **Deliverable:** Profiling report identifying top optimization opportunities

---

## Phase 4: Production Systems (Weeks 13-16)

*Goal: Build complete production-grade AI systems*

### Week 13: RAG Pipeline — Retrieval Infrastructure

- Build GPU-accelerated semantic search with FAISS
- Embedding pipeline: batch encoding, index construction, similarity search
- Benchmark retrieval latency vs index size
- **Deliverable:** Working vector search system with throughput benchmarks

### Week 14: RAG Pipeline — Generation & Evaluation

- Integrate retrieval with vLLM generation end-to-end
- Measure end-to-end latency: retrieval + generation
- Evaluate answer quality with and without retrieval context
- **Deliverable:** Complete RAG system with latency and quality measurements

### Week 15: Multi-Model Routing & Orchestration

- Build request router: classify query complexity, route to 3B vs 14B model
- Measure routing accuracy and latency overhead
- Cost modeling: savings from routing vs single large model
- **Deliverable:** Adaptive routing system with cost analysis

### Week 16: Production Hardening

- Request queuing, timeout handling, graceful degradation under overload
- Health checks, circuit breakers, retry logic
- Load testing: find failure modes and recovery behavior
- **Deliverable:** Production-hardened inference service with reliability playbook

---

## Phase 5: Operations & Cost Modeling (Weeks 17-20)

*Goal: Build production operations and business modeling capabilities*

### Week 17: Infrastructure Cost Modeling

- Build TCO calculator for this hardware configuration
- Compare on-premise (4x RTX 3090) vs cloud alternatives (Lambda Labs, RunPod, AWS)
- Break-even analysis across usage patterns and model sizes
- **Deliverable:** Cost modeling framework with measured data from this hardware

### Week 18: Capacity Planning Framework

- Model throughput, memory, and latency constraints together
- Capacity planning under uncertainty: traffic spikes, model updates
- SLA definition and measurement: p99 latency budgets, error rate targets
- **Deliverable:** Capacity planning guide for LLM serving systems

### Week 19: Full Observability Stack

- Deploy Prometheus + Grafana dashboards for all active serving infrastructure
- Track SLA-relevant metrics (p50/p95/p99 latency, throughput, error rates)
- Simulate failure scenarios and recovery
- **Deliverable:** Production monitoring dashboard and reliability playbook

### Week 20: Latency-Quality Tradeoff Framework

- Document how quantization, batching, caching, and model size affect user experience
- Create decision framework for model selection (when to use 3B vs 7B vs 14B vs 70B)
- **Deliverable:** Model selection guide with measured data from this hardware

---

## Phase 6: Capstone & Portfolio (Weeks 21-24)

*Goal: Demonstrate end-to-end capability*

**Build one comprehensive project combining engineering and product thinking:**

### Recommended: "Enterprise Inference Platform"

- Multi-model serving with automatic routing based on query complexity
- GPU-accelerated semantic search for document retrieval
- Cost tracking per request/tenant
- Full observability stack with SLA monitoring
- Admin dashboard showing utilization, costs, latency distribution
- Production deployment with graceful degradation and auto-scaling

**Deliverable:** Full documentation including technical architecture diagram, performance benchmarks, cost analysis, and demo.

---

## Parallel Learning Streams

**Throughout all phases:**

- Read NVIDIA technical blogs and GTC talks (weekly, 1-2 hours)
- Write technical blog posts about learnings (bi-weekly)
- Review and update "AI Infrastructure Knowledge Map" (monthly)
- Practice explaining technical concepts in product terms (monthly)

---

## Key Changes from Original Plan

| Original Plan | Revised Plan | Reason |
|---|---|---|
| Weeks 1-2: TensorRT + CUDA-X benchmarks | Week 1: Transformers baselines, Week 2: TensorRT | Baselines needed first to have something to optimize against |
| Weeks 3-4: Multi-GPU + Nemotron-70B tensor parallelism | Week 3: Multi-GPU topology + data/pipeline parallelism, Week 4: vLLM single-GPU | Tensor parallelism unviable without NVLink; vLLM pulled forward |
| Weeks 5-6: Triton first | Weeks 5-6: vLLM multi-GPU + sustained load | vLLM continuation is natural; Triton follows in Weeks 7-8 |
| Weeks 7-8: vLLM | Weeks 7-8: Triton + framework comparison | vLLM already started; Triton and comparison work here instead |
| NVLink side experiments (separate track, Weeks 7-9) | NVLink TP=2 benchmark completed in Week 7; comparison integrated into Week 8 | Bridge arrived at start of Week 7; NVLink work absorbed into main curriculum rather than treated as a side project |
| Week 11-12: Custom CUDA Kernels | Week 11: Speculative decoding + KV cache compression, Week 12: NSight profiling | More production-relevant than writing custom kernels from scratch |
| Phase 5: Separate PM track | Phase 5: Integrated operations + cost modeling | Cost modeling benefits from having all benchmark data in hand |

---

*Training started: January 13, 2026*
*Current status: Week 7 complete, beginning Week 8*
*Hardware: 4x RTX 3090 (96GB total), Gigabyte B650 Eagle AX, Ubuntu 24.04, CUDA 12.6*
*NVLink bridge: Installed (AORUS GeForce RTX NVLink, GPU0+GPU2, NV4)*