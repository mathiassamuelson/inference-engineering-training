# Training Plan: NVIDIA Inference Stack Mastery (Revised)

*Last updated: April 7, 2026 — reflects actual progress through Week 8 (four-day arc)*

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

## Phase 2: Production Inference at Scale (Weeks 5-10)

*Goal: Master multi-GPU production serving, larger models, multi-model orchestration, and framework comparison. Phase extended from 4 to 6 weeks to accommodate the inserted four-day Gemma 4 deployment work and the Triton/framework comparison content originally planned for Week 8.*

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

### Week 8: Gemma 4 Day-1 Deployment Arc ✅

*Inserted week. Gemma 4 dropped on April 2, 2026; the curriculum was paused for four days of focused deployment work against this hardware before resuming the original Triton content. The four days form a single connected arc — each day's findings shaped the next day's experiment.*

#### Day 1 (April 3) — Gemma 4 31B Dense initial deployment, segfault discovery

- Built llama.cpp from source with CUDA support; loaded Q8_0 GGUF (32.6 GB) of `ggml-org/gemma-4-31B-it-GGUF` across NVLink pair (GPU 0+2) via layer splitting
- Documented Gemma 4's hybrid attention architecture from startup logs: 50 sliding-window layers (1024-token window) + 10 global full-context layers; KV head count varies per layer (16 SWA / 4 global) for memory efficiency
- llama.cpp auto-fit context from architectural 262K to 104,704 tokens to match available VRAM; total VRAM ~22.8 GiB per card
- Built custom Jinja chat template `gemma4-no-think.jinja` to disable thinking mode; **3.6x reduction in response time** (8.3s → 2.3s) on identical prompts with identical generation speed — savings come entirely from not generating ~200 wasted internal-reasoning tokens
- Discovered prompt-length segfault: model crashes processing prompts above ~5,400 tokens. Binary-searched the boundary (5,482 works, ~5,600 crashes). Reproduced on both single-GPU and multi-GPU configurations. Drafted clean reproduction case and bug report for the llama.cpp GitHub repo
- Attempted vLLM with `--quantization fp8` (W8A16 Marlin) as a fallback path; OOM'd during sampler warmup. Reduced `max-model-len`, `max-num-seqs`, enabled `--enforce-eager`, attempted `--kv-cache-dtype fp8` — every variation OOM'd. Root cause: vLLM's FP8 weight-only path produced ~16.5 GiB per GPU, ~1 GiB more than llama.cpp's Q8_0 split, and consumer 24GB cards have no headroom for that delta
- **Blocker for statmon-ai:** the 5.4K token crash limit is incompatible with statmon-ai's ~6K system prompt + ~7K–15K total context, making this configuration unusable for the real workload pending the upstream fix
- **Key finding:** Day-1 model deployments are rough on consumer hardware — both major frameworks had Gemma 4 support added within hours of release, both had issues, neither was production-ready on day one

#### Day 2 (April 4) — llama.cpp bug fix, clean throughput sweep, consumer P2P topology lesson

- llama.cpp maintainers turned around the segfault fix overnight. Pulled latest, rebuilt, reran sweep — clean run end-to-end including past the old crash boundary. An impressive turnaround for a day-1 model support bug
- First sweep produced an anomaly: target=4K row reported 547 tok/s prefill vs ~900 tok/s neighbors. Reproduced exactly on rerun, ruling out variance. Investigation led to a methodology lesson:
  - llama-server's slot cache reuse was matching common prefixes between consecutive prompts in the benchmark (which used identical framing + variable filler), so the server was only prefilling new tokens at the end while the script reported throughput against the (smaller) `timings.prompt_n` work counter rather than the (larger) true `usage.prompt_tokens` size
  - llama-server's OpenAI-compatible endpoint **silently drops `cache_prompt: false`** — that field is a native llama.cpp extension on `/completion`, not `/v1/chat/completions`. No error, no warning
  - Working fix: prepend a UUID nonce to each prompt, defeating cache reuse at the content level. Reran with clean cold-prefill measurements throughout
- Clean throughput results: prefill rises 862 → 1,173 tok/s (plateaus from 14K context), decode 23.9 → 20.4 tok/s (14.6% drop across 54x context span, weight-bandwidth bound throughout). **The corrected prefill ceiling is ~18% higher than the original artifact-laden measurement** — meaningful for TTFT planning
- **NVLink vs PCIe comparison** (`CUDA_VISIBLE_DEVICES=0,2` vs `0,1`): predicted ~1.5% prefill hit at 28K from naive bandwidth math; measured 21% hit (peaking at 30% mid-range). Decode unaffected. Order-of-magnitude prediction miss
- Diagnosis with `nvidia-smi topo -p2p r`: only the NVLink pair (GPU 0+2) reports `OK` for peer-to-peer; all other pairs report `CNS` (chipset not supported). When CUDA can't enable peer access, it silently falls back to staging through pinned host memory: GPU → RAM → GPU. That two-hop path serializes with host operations, adds CUDA stream sync per transfer, and gets paid every prefill chunk (~14 transfers for a 28K prompt at default chunk size)
- **Key finding:** Peer-to-peer DMA between consumer GPUs is not guaranteed by topology alone. On motherboards where additional GPU slots hang off the chipset rather than the CPU's own PCIe root complex, CUDA denies peer access and falls back to host staging — and the cost is much larger than nominal bandwidth math predicts. **`nvidia-smi topo -p2p r` is the diagnostic that would have flagged this before purchase.** Drafted long-form LinkedIn article covering the day-1 deployment story, hybrid attention architecture, no-think template, and the consumer topology lesson with the diagnostic command as the main takeaway

#### Day 3 (April 5) — Gemma 4 26B-A4B MoE on llama.cpp, head-to-head with the dense sibling

- Downloaded `gemma-4-26B-A4B-it-Q8_0.gguf` and launched with the same llama.cpp parameters as Day 2's dense run, changing only the model file
- Architectural inspection from startup logs: MoE is structurally a half-depth, half-width version of the dense model with MoE FFN blocks (128 experts, 8 routed per token, expert FFN dim 704) replacing dense FFN. 30 layers vs 60. 25/5 SWA/global ratio mirrors dense's 50/10 exactly. KV heads halved on global layers (2 vs 4)
- **Memory result was a categorical difference, not an incremental one:** dense had to shrink to 104K context to fit; **MoE loaded the full architectural 262K context window with 11.4 GB free VRAM remaining**. The mechanism is per-cell KV cost — MoE has half the layers, half the global KV heads → roughly a quarter the cost per token per layer. 2.5x more context cells fit in half the KV cache memory
- Throughput results, NVLink sweep: decode 112 → 94 tok/s across context, prefill peaks at 4,276 tok/s at 14K and holds ~4,200 tok/s at 28K. **Decode is a flat 4.6–4.7x multiplier over dense at every context length; prefill is 2.6x at short context growing to 3.6x at long context** (FFN-bound prefill, hybrid attention keeps it linear-ish)
- Made a wrong-magnitude prediction about MoE PCIe penalty (predicted 30–45% based on transfer-to-compute ratio argument; measured 24%) — directionally right, badly overshooting in magnitude. Lesson noted: predict-before-measure discipline catches mechanism reasoning errors even when the prediction is wrong
- **New finding from PCIe sweep — MoE exposes decode latency that dense hid:** consistent ~6% PCIe penalty on MoE decode at every context length, where dense decode was noise-level unaffected. The activation payload is trivial either way; the mechanism is fixed per-transfer overhead (latency, kernel launch, sync) becoming a larger fraction of the shorter per-token budget (~10ms MoE vs ~48ms dense). **General principle: accelerating compute exposes communication costs that were previously hidden.** This will matter more, not less, as future models push active params down while keeping total params high
- Added `--model-name` parameter to `exp2_throughput_sweep.py` — name now flows through request payload, summary table header, JSON metadata, and default output filename. Removed the previously-hardcoded `context_limit: 104704` from JSON metadata since it was specific to the dense run
- **Statmon-ai approach pivot.** Today's MoE results combined with an honest reread of the statmon-ai system prompt forced a significant change in the planned application approach:
  - The efficiency argument for fine-tuning to eliminate the system prompt has collapsed: MoE's 3.6x prefill advantage reduces the ~5K prompt's prefill cost from ~5s to ~1.25s, and llama.cpp's default prefix caching pays that cost once per session rather than per request. Amortized per-request cost approaches zero
  - Audit of the statmon-ai prompt revealed it's ~75% reference documentation (CLI commands, enum values, filter syntax), ~10% tool descriptions, ~20% behavioral playbooks, ~5% role framing. **Reference material is exactly what fine-tuning handles worst** — models hallucinate enum values, substitute plausible-but-wrong command names, drift out of sync when the underlying system changes
  - **Revised plan:** deploy 26B-A4B MoE in production with the full existing system prompt and prefix caching enabled. Decouple the fine-tuning learning goal from statmon-ai entirely; pursue it later on a smaller model with a cleaner task (output formatting, behavioral patterns) where fine-tuning's strengths actually apply
- LinkedIn Pulse article drafted: head-to-head MoE vs dense with throughput and memory tables, architectural framing, PCIe topology replication, and a deliberately-hedged section on multi-user deployment implications (no concurrent-load benchmark was run, so claims are inferred not measured)
- **Key finding:** "Deployment category" is a more useful frame than raw throughput. The dense and MoE look similar on a spec sheet but belong to different deployment categories on this hardware: dense is a single-user desktop deployment with cramped context; MoE is a plausible small-team shared inference target with the full context window and substantial VRAM headroom

#### Day 4 (April 7) — Gemma 4 26B A4B MoE on vLLM TP=2 over NVLink, six failure chain

- Set out to deploy the same MoE model under vLLM with tensor parallelism, both as a framework comparison against Day 3's llama.cpp numbers and to characterize concurrency behavior
- Worked through six distinct day-1 deployment failure modes before landing on a working configuration:
  1. **Pip dependency resolver wall** — vLLM nightly that satisfied all constraints predates the Gemma 4 merge; pivot to the official `vllm/vllm-openai:gemma4` Docker image
  2. **protoLabsAI FP8 block-shape mismatch** — community pre-quantized FP8 checkpoint requires `block_k=128`, but Gemma 4's MLP intermediate dim 2112 isn't divisible (1056 per shard at TP=2, still not divisible). Structural incompatibility, would have failed at TP=1 too. Pivot to canonical BF16 + on-the-fly FP8
  3. **FP8 KV cache hardware incompatibility** — Inductor codegen requires hardware-level `bf16 → fp8e4nv` cast, which Ampere SM 8.6 doesn't have (FP8 instructions arrived with Ada/Hopper SM 8.9+). Pivot to bf16 KV cache
  4. **Marlin FP8 MoE shape table miss** — kernel returns sentinel error for K=352 (Gemma 4's expert FFN intermediate 704 sharded TP=2). Pivot to `--moe-backend triton`
  5. **Triton FP8 MoE quant scheme mismatch** — Triton MoE handles arbitrary K but doesn't accept the per-tensor static-weight × per-tensor dynamic-activation FP8 scheme that vLLM's online quantization produces
  6. **`batched_triton` not user-selectable** — listed in the auto-selection candidate set but not exposed via CLI. Of the user-selectable backends, all others are AMD/Hopper-only or already-tried. End of FP8 backend road on Ampere
- **Pivot to AWQ-INT4** via `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` — completely different code path through `compressed-tensors`, INT4 Marlin kernels with different shape support tables. Loaded cleanly, sanity request returned coherent text
- Two concurrency measurements taken:
  - `max-model-len 16384`: **24.24x concurrent users**
  - `max-model-len 262144`: **3.91x concurrent users at the full architectural context window**
- **The 262K configuration is not achievable on llama.cpp on this hardware** — Day 1's dense run auto-shrunk to 104K, and even Day 3's MoE on llama.cpp at 262K had less per-request concurrency headroom than vLLM's PagedAttention provides. The framework decision matrix gains a real data point
- Persistent warning in the working config: `Using default MoE config. Performance might be sub-optimal! Config file not found at .../E=128,N=352,device_name=NVIDIA_GeForce_RTX_3090,dtype=int4_w4a16.json`. Same K=352 shape that broke Marlin FP8 MoE in Failure 4 also lacks a tuned config in the INT4 W4A16 table; here it falls back to a generic default config and runs (warning, not error) — but throughput is sub-optimal vs what's theoretically achievable on this shape
- Single-request decode rate from vLLM's 10s windowed averages worked out to ~10–14 tok/s — meaningfully slower than llama.cpp's ~94–112 tok/s on Day 3 against the same model. The untuned MoE kernel is the most likely explanation; per-completion timing instrumentation needed for precise measurement
- **Key findings:** (1) FP8 weight quantization and FP8 KV cache have different hardware requirements — Marlin emulates weights anywhere via upconversion, KV cache FP8 needs Ada/Hopper for the hardware cast; (2) hybrid attention's KV savings produce a 2.6x effective concurrency benefit at long context vs naive predictions (24.24/3.91 = 6.2x ratio observed vs naive 16x); (3) day-1 MoE deployment requires walking through multiple kernel/quant compatibility layers — six distinct rejection points, not a single failure

**Models tested across the full week:** Gemma 4 31B Dense (llama.cpp Q8_0, layer-split TP), Gemma 4 26B A4B (llama.cpp Q8_0 layer-split, vLLM AWQ-INT4 TP=2 NVLink, attempted vLLM FP8 in multiple configurations)

**Frameworks exercised:** llama.cpp (b8660 → post-fix → b8664), vLLM 0.18.2rc1.dev73 via Docker

**Repository deliverables:**
```
phase-2-production/week-08/
├── exp2_throughput_sweep.py              # Generalized with --model-name (Day 3)
├── results/
│   ├── exp2_throughput_sweep_gemma4-31B-it_nvlink.json    # Day 2
│   ├── exp2_throughput_sweep_gemma4-31B-it_pcie.json      # Day 2
│   ├── exp2_throughput_sweep_gemma4-26B-A4B-it_nvlink.json   # Day 3
│   └── exp2_throughput_sweep_gemma4-26B-A4B-it_pcie.json     # Day 3
└── (no scripts created Day 4 — server-side configuration only)

docs/linkedin/
├── 2026-04-04-gemma4-31b-rtx3090.md      # Day 2 article
└── 2026-04-05-gemma4-26ba4-vs-31b.md     # Day 3 article (head-to-head)
```

**Custom tooling carried forward:** `gemma4-no-think.jinja` template (Day 1), bug report draft for llama.cpp `cache_prompt` silent-drop on `/v1/chat/completions` (Day 2 follow-up)

### Week 9: Gemma 4 MoE Optimization & Framework Comparison

*Continuation of Week 8 — focused on extracting comparative throughput data and contributing back upstream where possible. Original Week 9 content (quantization methods) pushed to Week 11.*

- **Generalize the throughput sweep script for vLLM endpoint compatibility.** Day 3 added `--model-name` parameterization, but the script targets llama.cpp's request/response shape. Validate or adapt for vLLM's OpenAI-compatible endpoint (response field names, completion token reporting). Single script targets both backends with identical CLI surface
- **Per-completion timing instrumentation** — replace vLLM's 10-second windowed averages with proper per-request prefill and decode rate measurement. Day 4's open question
- **Throughput sweeps against the AWQ vLLM configuration:** single-request sweep matching Day 3's llama.cpp matrix (same prompt sizes, same context windows) so the comparison is apples-to-apples. Document the untuned MoE kernel asterisk on all results
- **Concurrent-load benchmark against both backends.** Day 3 explicitly deferred this — the multi-user framing in the Day 3 LinkedIn Pulse article was hedged because no concurrent throughput was measured. Week 9 fills that gap. vLLM's continuous batching vs llama.cpp's slot-based concurrency at multiple concurrency levels
- **vLLM vs llama.cpp head-to-head analysis** combining Day 3's existing single-request llama.cpp data with Week 9's new vLLM data (single-request) and new concurrent data (both backends). The hypothesis to test: llama.cpp wins on single-request decode (mature MoE path), vLLM wins on concurrent throughput (continuous batching) — but the magnitude on each axis is the actual deliverable
- **MoE kernel tuning attempt:** run vLLM's MoE autotuner against the missing `(E=128, N=352, device_name=NVIDIA_GeForce_RTX_3090, dtype=int4_w4a16)` shape, benchmark against the default fallback, and quantify the throughput gap. If meaningful, this becomes the Week 9 stretch deliverable
- **Tuned config PR (stretch goal):** if the autotuner produces a meaningful improvement, prepare and submit a PR contributing the missing config to vLLM upstream. This would benefit anyone running Gemma 4 26B A4B on RTX 30-series hardware
- **vLLM Marlin FP8 MoE bug report (stretch goal):** file an issue with the clean reproduction case from Week 8 Day 4 documenting the K=352 shape gap in the FP8 Marlin MoE tuned-config table
- **Deliverable:** Framework comparison report with measured data from both backends across single-request and concurrent regimes, MoE tuning results, and (optimistically) merged or in-review upstream contributions
- **Models tested:** Gemma 4 26B A4B on both vLLM (AWQ-INT4) and llama.cpp (Q8_0)

### Week 10: Triton Inference Server Deep Dive & Framework Selection

*Original Week 8 content, pushed forward by the four-day Gemma 4 arc in Week 8 and the focused continuation in Week 9.*

- **Triton multi-model serving:** deploy embedding model + generation model simultaneously across GPUs, measure operational overhead
- **Dynamic batching optimization:** systematic tuning of batch sizes, queue delays, and instance counts
- **Prometheus metrics integration:** production observability for multi-model Triton deployment
- **NVLink TP=2 vs 4-GPU data parallel comparison:** NVLink 14B (2 GPUs) vs data parallel 3B (4 GPUs) — direct quality vs. concurrency tradeoff measurement
- **Framework decision matrix (now three-way):** when to use Triton vs vLLM vs llama.cpp vs combinations, informed by Week 9's measured comparison data
- **Deliverable:** Multi-model inference API with performance monitoring + framework selection guide with measured data across all three backends

---

## Phase 3: Optimization & Quantization (Weeks 11-14)

*Goal: Master model compression and optimization techniques. Shifted forward by 2 weeks to accommodate the Gemma 4 deployment work in Weeks 8-9 and the Triton work in Week 10.*

### Week 11: Quantization Methods (AWQ, GPTQ)

- Apply AWQ and GPTQ quantization to Llama 3.2 3B and a 7-14B model
- Measure quality degradation: perplexity, generation coherence, task accuracy
- Benchmark INT4/INT8 vs FP16: throughput, latency, memory savings
- **Note:** Week 8 Day 4's AWQ-INT4 deployment of Gemma 4 26B A4B already established that AWQ via `compressed-tensors` works cleanly on Ampere and provides substantial memory savings. Week 11 focuses on the systematic quality measurement that Week 8 deferred
- **Deliverable:** Quality vs performance tradeoff analysis across quantization methods

### Week 12: Quantized Model Serving with vLLM

- Serve quantized models through vLLM: measure end-to-end production impact
- Capacity planning with quantization: how many more concurrent users per GPU?
- **Deliverable:** Quantization deployment guide with measured quality/performance tradeoffs

### Week 13: Speculative Decoding & KV Cache Compression

- Implement speculative decoding with a draft model (Llama 3.2 1B → 3B)
- Measure acceptance rate, latency improvement, and throughput impact
- KV cache compression techniques: sliding window, prefix caching
- **Deliverable:** Speculative decoding analysis with acceptance rate breakdown

### Week 14: NSight Profiling & Bottleneck Analysis

- Profile inference workloads with NVIDIA NSight Systems and NSight Compute
- Identify compute vs memory bandwidth bottlenecks at the kernel level
- Profile vLLM's PagedAttention and continuous batching kernel execution
- **Deliverable:** Profiling report identifying top optimization opportunities

---

## Phase 4: Production Systems (Weeks 15-18)

*Goal: Build complete production-grade AI systems*

### Week 15: RAG Pipeline — Retrieval Infrastructure

- Build GPU-accelerated semantic search with FAISS
- Embedding pipeline: batch encoding, index construction, similarity search
- Benchmark retrieval latency vs index size
- **Deliverable:** Working vector search system with throughput benchmarks

### Week 16: RAG Pipeline — Generation & Evaluation

- Integrate retrieval with vLLM generation end-to-end
- Measure end-to-end latency: retrieval + generation
- Evaluate answer quality with and without retrieval context
- **Deliverable:** Complete RAG system with latency and quality measurements

### Week 17: Multi-Model Routing & Orchestration

- Build request router: classify query complexity, route to 3B vs 14B model
- Measure routing accuracy and latency overhead
- Cost modeling: savings from routing vs single large model
- **Deliverable:** Adaptive routing system with cost analysis

### Week 18: Production Hardening

- Request queuing, timeout handling, graceful degradation under overload
- Health checks, circuit breakers, retry logic
- Load testing: find failure modes and recovery behavior
- **Deliverable:** Production-hardened inference service with reliability playbook

---

## Phase 5: Operations & Cost Modeling (Weeks 19-22)

*Goal: Build production operations and cost modeling capabilities*

### Week 19: Infrastructure Cost Modeling

- Build TCO calculator for this hardware configuration
- Compare on-premise (4x RTX 3090) vs cloud alternatives (Lambda Labs, RunPod, AWS)
- Break-even analysis across usage patterns and model sizes
- **Deliverable:** Cost modeling framework with measured data from this hardware

### Week 20: Capacity Planning Framework

- Model throughput, memory, and latency constraints together
- Capacity planning under uncertainty: traffic spikes, model updates
- SLA definition and measurement: p99 latency budgets, error rate targets
- **Deliverable:** Capacity planning guide for LLM serving systems

### Week 21: Full Observability Stack

- Deploy Prometheus + Grafana dashboards for all active serving infrastructure
- Track SLA-relevant metrics (p50/p95/p99 latency, throughput, error rates)
- Simulate failure scenarios and recovery
- **Deliverable:** Production monitoring dashboard and reliability playbook

### Week 22: Latency-Quality Tradeoff Framework

- Document how quantization, batching, caching, and model size affect user experience
- Create decision framework for model selection (when to use 3B vs 7B vs 14B vs 70B)
- **Deliverable:** Model selection guide with measured data from this hardware

---

## Phase 6: Capstone & Portfolio (Weeks 23-26)

*Goal: Demonstrate end-to-end capability*

**Build one comprehensive project combining the engineering threads from earlier phases:**

### "Enterprise Inference Platform"

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

---

## Key Changes from Original Plan

| Original Plan | Revised Plan | Reason |
|---|---|---|
| Weeks 1-2: TensorRT + CUDA-X benchmarks | Week 1: Transformers baselines, Week 2: TensorRT | Baselines needed first to have something to optimize against |
| Weeks 3-4: Multi-GPU + Nemotron-70B tensor parallelism | Week 3: Multi-GPU topology + data/pipeline parallelism, Week 4: vLLM single-GPU | Tensor parallelism unviable without NVLink; vLLM pulled forward |
| Weeks 5-6: Triton first | Weeks 5-6: vLLM multi-GPU + sustained load | vLLM continuation is natural; Triton follows later |
| Weeks 7-8: vLLM | Week 7: NVLink TP, Week 8: originally Triton | Bridge arrived at start of Week 7; NVLink work absorbed into main curriculum |
| Week 8: Triton + framework comparison | Week 8: four-day Gemma 4 deployment arc (inserted), Week 10: Triton + framework comparison | Gemma 4 dropped April 2, 2026 — paused curriculum to deploy on day one against this hardware. Each day's findings shaped the next; the four-day length emerged from the work, not from advance planning |
| Week 9: Quantization methods | Week 9: Gemma 4 MoE optimization, framework comparison, MoE tuned-config PR attempt | Day 4's open questions warranted a focused continuation week to extract head-to-head comparison data and attempt the upstream contribution |
| Phase 3 starts at Week 9 | Phase 3 starts at Week 11 | 2-week shift accommodates the Gemma 4 work in Weeks 8-9 and the Triton work in Week 10 |
| Week 11-12: Custom CUDA Kernels | Week 13: Speculative decoding + KV cache compression, Week 14: NSight profiling | More production-relevant than writing custom kernels from scratch |
| Phase 5: Separate PM track | Phase 5: Integrated operations + cost modeling | Cost modeling benefits from having all benchmark data in hand |
| 24-week program | **26-week program** | The four-day Gemma 4 arc and the Week 9 continuation extend the timeline by two weeks; later phases preserved at original length rather than compressed |

---

*Training started: January 13, 2026*
*Current status: Week 8 complete (4 days), beginning Week 9*
*Hardware: 4x RTX 3090 (96GB total), Gigabyte B650 Eagle AX, Ubuntu 24.04*
*NVLink bridge: Installed (AORUS GeForce RTX NVLink, GPU0+GPU2, NV4)*
