# Training Plan: NVIDIA Stack → AI Engineering/Product Management (Revised)

*Last updated: February 24, 2026 — reflects actual progress through Week 4*

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
- **Key finding:** Data parallelism is the only viable multi-GPU strategy on this hardware. PCIe x1 has zero impact on inference once models are GPU-resident. Tensor parallelism requires NVLink
- **Hardware:** 4x RTX 3090, Gigabyte B650 Eagle AX topology mapped

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
- **Key finding:** Framework architecture dominates all other optimization axes — more impactful than FP16 optimization (1.56x), TensorRT conversion, or multi-GPU scaling. vLLM's operational qualities (request queuing, continuous batching, graceful degradation) proved stable and production-ready under sustained load
- **Models tested:** Llama 3.2 3B Instruct (multi-GPU), Mistral 7B Instruct v0.3 (single GPU)

### Week 6: Larger Model Scaling & Triton Introduction

- Scale to larger models across multiple GPUs: Qwen 2.5 14B across 2 GPUs, measure whether the 7x framework advantage grows with model size
- Compare single-GPU 7B serving vs multi-GPU 14B serving: throughput, latency, cost-per-token tradeoffs
- Begin Triton Inference Server setup: install, configure model repository, deploy a non-LLM model (embedding or classification) to learn Triton fundamentals separately from vLLM
- **Deliverable:** Model size scaling analysis answering "at what model size does vLLM's advantage grow beyond 7x?" and initial Triton deployment

### Week 7: Triton Inference Server

- Deploy Triton with model repository (embedding model + generation model)
- Dynamic batching configuration and measurement
- Multi-model serving: simultaneous models on different GPUs
- Prometheus metrics integration for monitoring
- **Deliverable:** Multi-model inference API with performance monitoring

### Week 8: Triton vs vLLM Comparison & Framework Decision Matrix

- Head-to-head comparison: Triton vs vLLM for LLM serving (latency, throughput, operational features)
- Triton model ensembles: preprocessing → generation → postprocessing pipeline
- Build framework decision matrix: when to use Triton vs vLLM vs both (Triton + vLLM backend)
- **Deliverable:** Framework selection guide with measured data from this hardware

### NVLink Side Experiments (When Bridge Arrives)

- Re-run Week 3 topology benchmark: NVLink bandwidth vs PCIe
- Tensor parallelism within vLLM across GPU 0+1 NVLink pair
- Compare: NVLink tensor parallel vs single GPU vs data parallel
- Load 70B model (Nemotron or Llama 3.1 70B) across NVLink pair + remaining GPUs
- **Deliverable:** NVLink vs PCIe analysis, tensor parallelism viability assessment

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
- Fit larger models on single GPU via quantization (14B quantized vs 3B full precision)
- **Deliverable:** Quantization deployment guide with production capacity numbers

### Week 11: Model Optimization Deep Dive

- TensorRT-LLM integration with vLLM (if CUDA compatibility resolved)
- Speculative decoding: use small draft model to accelerate large model generation
- KV cache compression techniques
- **Deliverable:** Advanced optimization techniques benchmark

### Week 12: Profiling & Performance Analysis

- NSight Compute profiling of inference workloads
- Identify remaining bottlenecks: memory bandwidth utilization, kernel efficiency
- Compare profiling results across frameworks (transformers vs vLLM vs Triton)
- **Deliverable:** Performance profiling report with optimization recommendations

---

## Phase 4: AI Engineering Projects (Weeks 13-16)

*Goal: Build portfolio-worthy applications using skills from Phases 1-3*

**Choose 2 of 3 projects:**

### Project A: RAG System with GPU-Accelerated Vector Search

- Implement semantic search over technical documentation
- Use embedding model on one GPU, generation model on others
- Deploy with vLLM or Triton + vector DB (Milvus/Qdrant)
- **Deliverable:** Working RAG API with sub-100ms retrieval

### Project B: Multi-Model Routing Service

- Automatic query routing based on complexity (small model for simple queries, large model for complex)
- Cost tracking per request
- A/B testing framework for model comparison
- **Deliverable:** Intelligent model router with cost/quality tradeoff dashboard

### Project C: Fine-tuning Pipeline

- Fine-tune Llama 3.2 3B on domain-specific data using LoRA/QLoRA
- Compare base vs fine-tuned model performance
- Deploy fine-tuned model through production serving pipeline
- **Deliverable:** End-to-end fine-tuning and deployment pipeline

---

## Phase 5: Production Operations & Cost Modeling (Weeks 17-20)

*Goal: Translate technical skills into production and product insights*

### Week 17-18: Infrastructure Cost Modeling

- Build TCO calculator for inference deployments
- Compare: 4x RTX 3090 rig vs cloud alternatives (Lambda Labs, RunPod, AWS)
- Break-even analysis for different usage patterns
- **Deliverable:** Interactive cost calculator with real benchmarks from this hardware

### Week 19: Observability & Reliability

- Full monitoring stack (Prometheus + Grafana)
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
| NVLink not in original plan | NVLink side experiments when bridge arrives (~Weeks 7-9) | Hardware constraint discovered in Week 3 |
| Week 11-12: Custom CUDA Kernels | Week 11: Speculative decoding + KV cache compression, Week 12: NSight profiling | More production-relevant than writing custom kernels from scratch |
| Phase 5: Separate PM track | Phase 5: Integrated operations + cost modeling | Cost modeling benefits from having all benchmark data in hand |

---

*Training started: January 13, 2026*
*Current status: Week 4 complete, beginning Week 5*
*Hardware: 4x RTX 3090 (96GB total), Gigabyte B650 Eagle AX, Ubuntu 24.04, CUDA 12.6*
*NVLink bridge: On order, ETA ~Weeks 7-9*