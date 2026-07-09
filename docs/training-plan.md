# Training Plan: NVIDIA Inference Stack Mastery (Revised)

*Last updated: July 5, 2026 — **Conclusion revision**. The program concludes at 16 weeks. Weeks 15–16 are redefined from the speculative-decoding and NSight-profiling content to the program's two concluding weeks: Week 15 delivers the operational proof (cross-tier interference characterization + the delegation-architecture write-up) on the frozen v0.23.0 / 4×3090 production configuration; Week 16 consolidates and concludes (repo renames, journal consolidation, capstone summary, method Pulse). Phases 4–6 (Weeks 17–28) are dispositioned rather than executed — achieved in substance, migrated to the successor program, or deferred to a potential follow-on inference module — see the Disposition section and the Key Changes log.*

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

## Phase 2: Production Inference at Scale (Weeks 5-10) ✅ Complete

*Goal: Master multi-GPU production serving, larger models, multi-model orchestration, and framework comparison. Phase extended from 4 to 6 weeks to accommodate the inserted four-day Gemma 4 deployment work (Week 8), the Gemma 4 MoE optimization continuation (Week 9), and the inference-reference-stack work (Week 10).*

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

*Inserted week. Gemma 4 dropped on April 2, 2026; the curriculum was paused for four days of focused deployment work against this hardware before resuming.*

- Day 1: Gemma 4 31B Dense Q8_0 deployment, segfault discovery, vLLM FP8 OOM exploration
- Day 2: NVLink characterization for pipeline parallelism on 31B Dense — ~1,170 tok/s prefill plateau, ~20-24 tok/s decode; PCIe P2P topology gotcha discovered (CNS chipset-not-supported, 21-30% prefill penalty)
- Day 3: Gemma 4 26B A4B MoE deployment on llama.cpp Q8_0 — measured concurrency, deployment-category framing
- Day 4: Gemma 4 26B A4B MoE on vLLM with AWQ-INT4 after six distinct day-1 deployment failures; framework decision matrix gains real data
- **Key finding:** "Deployment category" is a more useful frame than raw throughput. The dense and MoE look similar on a spec sheet but belong to different deployment categories on this hardware: dense is a single-user desktop deployment with cramped context; MoE is a plausible small-team shared inference target with the full context window and substantial VRAM headroom
- **Models tested:** Gemma 4 31B Dense (llama.cpp Q8_0), Gemma 4 26B A4B MoE (llama.cpp Q8_0 + vLLM AWQ-INT4)

### Week 9: Gemma 4 MoE Optimization & KV Sizing Investigation ✅

*Continuation week. Day 4's open questions from Week 8 warranted focused experimentation.*

- Day 1: vLLM single-request throughput sweep on 26B MoE AWQ-INT4 (TP=2 NVLink) — baseline data
- Day 2: llama.cpp single-request throughput sweep on 26B MoE Q4_K_M (layer-split PP=2 NVLink); methodology reframe to "TP vs layer-splitting on asymmetric interconnect" rather than "framework A vs framework B"
- Day 3: Diagnosed vLLM KV cache sizing bug, reproduced on two architectures, reported upstream as [vllm-project/vllm#39133](https://github.com/vllm-project/vllm/issues/39133) with two-architecture reproduction case. Original Day 4 concurrent benchmarking explicitly paused — running it against the buggy vLLM build would produce real numbers with provisional interpretation
- *Epilogue (May 17): re-test against vLLM 0.21.0 HMA fix.* Re-measured single-request throughput on 26B MoE AWQ-INT4 with the Hybrid Memory Allocator landed. KV pool capacity at MML=262144 went from ~95K tokens to ~891K tokens — 9.3× increase with no VRAM budget change. Analytical model fit identified a remaining ~2× over-allocation on global layers (K=V unification not yet applied) and a ~400 MiB fixed per-sequence overhead. Single-request throughput essentially unchanged across the fix — the bug was in allocator bookkeeping, not attention math
- **Key findings:** (1) PCIe topology produces a ~21-30% prefill penalty via host-staging that naive bandwidth math doesn't predict; (2) vLLM pre-0.21 used ~5.5× more KV memory per token of capacity than llama.cpp on the same model; (3) the #39133 bug inverted apparent architectural conclusions because TP looked worse than it actually was; (4) HMA landing in vLLM 0.21.0 captures most but not all of the available KV efficiency
- **Models tested:** Gemma 4 26B A4B MoE on both vLLM (AWQ-INT4) and llama.cpp (Q4_K_M)

### Week 10: Inference Reference Stack — partial ⚠️ (paused as side-quest)

*Pivoted at session start from the originally-planned Triton + framework comparison. The Triton path was blocked by the NGC Triton vLLM image bundling vLLM 0.15.1, three minor versions behind what the Week 9 work required. The pivot: build a public reference deployment repository (`inference-reference-stack`) using vLLM's built-in OpenAI-compatible server, fronted by nginx, with Prometheus + Grafana observability — the architectural concerns above the inference engine being the actual learning target.*

- Day 1 ✅ vLLM in Docker Compose, public repository scaffolded with Apache 2.0 license, vLLM image pinned by digest
- Day 2 ✅ Prometheus + Grafana + dcgm-exporter wired up; starter dashboard with KV cache, latency histograms, GPU metrics. Grafana bound to LAN; vLLM/Prometheus on loopback
- **Days 3-5: deferred as side-quest.** nginx reverse proxy with TLS termination, API key authentication, per-token metering, and Slack-integrated alerting remain on the roadmap. They are now opportunistic returns rather than scheduled training-plan work
- **Key finding:** The architectural value of a production-pattern inference stack lives above the inference engine. Building the observability layer with metric-name-version-binding pitfalls, GPU/KV/latency dashboards, and the Compose/network topology was the high-density learning. The remaining nginx/TLS/API key work is real but well-documented elsewhere and not unique to AI infrastructure; returning to it as time permits is a better trade than completing it serially before Week 11
- **Repository:** [`inference-reference-stack`](https://github.com/<owner>/inference-reference-stack) (public)
- **Models tested:** Gemma 4 26B A4B MoE (`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`) on vLLM with the production-pattern stack

---

## Phase 3: Optimization, the Delegation Architecture & Program Conclusion (Weeks 11-16)

*Goal: Master parallelism strategies, then validate and operationalize the tiered delegation architecture that emerged from the Week 11 parallelism work. Phase 3 starts at Week 11 and runs 6 weeks. Weeks 12–13 were repurposed from the original quantization-methods content (rendered largely redundant by the Gemma deployment work in Weeks 8/9/11) to sub-agent-tier validation and two-tier QAT quality characterization; Week 14 is a Phase-3 close-out / loose-ends week. Weeks 15–16, originally speculative decoding and NSight profiling, are redefined as the program's conclusion — the operational proof (Week 15) and the capstone consolidation (Week 16). Phase 3 is the program's final phase; the deferred optimization topics and the quality-degradation measurement are recorded in the Phase 4–6 disposition below.*

### Week 11: Parallelism Strategy Comparison — TP vs PP on Gemma 4 31B Dense FP8 ✅

*The closing chapter of the parallelism-strategy thread that runs Week 3 (multi-GPU topology) → Week 7 (NVLink TP) → Week 8 Day 2 (NVLink PP characterization) → Week 9 Day 2 (paused mid-comparison due to #39133). With the HMA fix landed in vLLM 0.21.0, that thread can now close cleanly with controlled, single-framework data.*

**Outcome (6 days):** the parallelism question closed, and a max-context-length characterization (Day 6) surfaced the finding that reframed the project's direction.

- **KV cost model (Day 2):** TP=2 per-seq KV/GPU ≈ 1.97 GiB + 39.2 KiB/token × seq_len, validated to ~1% and used to predict every later ceiling. Text-only deployment drops the vision tower entirely (15.85 vs 16.93 GiB/GPU), raising available KV to 4.04 GiB/GPU.
- **PP=2 non-viable (Day 3):** the 256K-vocab embedding/LM-head don't shard under PP — they land whole on the end stages, starving KV. Serviceable context ~12× smaller than TP=2.
- **PP=4 viable but doesn't win where it counts (Days 4–5):** ~3.9× TP=2's KV pool, but ~1.7× slower decode (structural to pipelining over the host bridge, not steerable). Prefill crosses above TP=2 between 8K–16K. At c=4, TP=2 won aggregate throughput and fan-out completion at every prompt size — the "bigger pool → more throughput" thesis was refuted; capacity and throughput are distinct ceilings.
- **Max-MML ceilings (Day 6):** TP=2 is KV-bound at 54,496 tokens (util 0.95) / 66,848 (util 0.97, CUDA-graph tax recovered, +22.7%, crossing the 64K tier); PP=4 is *architecture*-bound at the full 256K with KV to spare. But PP=4 serves 256K at ~15 tok/s with ~5-minute TTFT.
- **The reframe:** for an interactive use case, fit is not the bar — usability is. PP=4's 256K is architecturally reachable but not interactively usable. **Neither single config serves the use case well** (TP=2 interactive but context-limited; PP=4 long-context but not interactive), which is the evidence that motivates the delegation architecture now anchoring Weeks 12–13.
- **Tooling:** `tools/throughput_sweep.py` (schema v3, concurrency-aware); `tools/start-vllm.sh` gained `--device-order` (Day 4) and `--profiler-cudagraphs {on,off}` (Day 6).
- **Publication:** held. The architectural finding (tiered delegation) is the strong, portable claim but is a hypothesis until the sub-agent tier is validated (Week 12). The TP-vs-PP data lacks generality without the use-case context; the CUDA-graph tax mis-advice is an upstream issue, not a post.
- **Models tested:** Gemma 4 31B Dense FP8 on vLLM (TP=2, PP=2, PP=4).

*Original forward-looking plan for the week, preserved for reference:*

- **Configuration:** Gemma 4 31B Dense, FP8 weights (`RedHatAI/gemma-4-31B-it-FP8-block`), vLLM 0.21.0, BF16 KV cache (Ampere FP8-KV is SM 8.9+, doesn't apply here)
- **Comparisons:**
  - TP=2 on NVLink pair (GPUs 0+2)
  - PP=2 on NVLink pair (GPUs 0+2)
  - PP=4 across all four GPUs (stretch — characterizes mismatched-interconnect ceiling; two of three boundary crossings hit PCIe x1)
- **What the strategies actually differ on:** Both TP=2 and PP=2 split weights and KV cache across the two GPUs — TP shards every layer's heads, PP partitions the layer range. Per-GPU memory footprints come out roughly comparable. The interesting differences are (a) **communication pattern** — TP all-reduces after every layer (60 sync points per forward pass), PP hands activations across the boundary once per pipeline-stage transition (1 boundary on PP=2, 3 on PP=4); (b) **compute concurrency** — TP runs both GPUs simultaneously on every token, PP runs them sequentially per sequence and only recovers GPU utilization with batching / continuous batching; (c) **decode vs prefill characteristics** — these communication patterns affect decode and prefill differently, and that's what the experiment will surface
- **Load regimes:** single-request throughput sweep matching prior weeks' methodology + concurrent throughput across configurable concurrency levels. Concurrent is especially important here because PP's compute-concurrency story emerges only under load
- **Methodology:** each configuration run at its natural settings rather than artificially matched. Empirical per-GPU KV pool capacity may differ across configs in ways the analytical model doesn't predict (CUDA graph buffers, per-pipeline-stage overhead, framework implementation quirks). Report per-GPU KV capacity as a column alongside throughput data so any differences are visible to a reader without forcing them to guess
- **Prerequisite work:** extend `tools/throughput_sweep.py` for concurrent load — configurable concurrency, per-request timings preserved, aggregate throughput computed. This is the gating engineering task and likely Day 1 of the week
- **Reference data:** Week 8 Day 2 has llama.cpp Q8_0 layer-split PP=2 numbers on this exact model (~1,170 tok/s prefill, ~20-24 tok/s decode); they're not direct comparators (different framework, different number system) but provide context
- **Open questions about vLLM 0.22.0:** released May 29, includes a Gemma 3/4 multi-GPU fix ([#42630](https://github.com/vllm-project/vllm/pull/42630)) and shared KV-cache layers ([#35045](https://github.com/vllm-project/vllm/pull/35045)). Default position is to stay on 0.21.0 for stability during the experiment; check #42630 relevance at kickoff. The 0.21 vs 0.22 comparison is itself a candidate post-Week-11 experiment if shared-KV-cache-layers closes the K=V gap identified May 17
- **Deliverable:** TP vs PP report with single-request and concurrent throughput, per-GPU KV capacity, communication-pattern analysis, PCIe topology cost characterization. LinkedIn Pulse candidate on the TP-vs-PP framing for the NVLink-pair-only audience (PP=4 is data-for-self, not for publication)
- **Models tested:** Gemma 4 31B Dense FP8 on vLLM (TP=2, PP=2, PP=4)

### Week 12: Sub-agent tier validation & the delegation architecture

*Repurposed from the original "Quantization Methods (AWQ, GPTQ)" block. Rationale: the Gemma work already established quantization as a working baseline — AWQ-on-Ampere (Week 8), KV-side savings under HMA (Week 9), FP8 datapoints (Week 11) — so the deployment-performance side of the original quantization weeks is redundant. The general operating principle is to run the highest-fidelity model that gives an acceptable context window, not to compare one quantization against another. The systematic quality-degradation measurement (the one part of the old Week 12 still wanted) moves to Week 24. This week instead validates the cheap tier of the delegation architecture that emerged from Week 11.*

*The architecture, stated substrate-neutrally: a capable model orchestrates and reasons over distilled findings; cheap, fast specialists do the bulk context work; the tiers are decoupled from the deployment substrate, so the same design runs on frontier APIs or self-hosted weights depending on a customer's cost and privacy constraints. The self-hosted realization here (31B orchestrator + 12B sub-agents on consumer GPUs) is the proof case, not the thesis.*

- **The load-bearing open question:** `google/gemma-4-12B-it-qat-w4a16-ct` was downloaded in Week 11 but failed to load on pinned vLLM 0.21.0 (`gemma4_unified` architecture unsupported). The 12B "Unified" model is encoder-free — it projects raw image/audio directly into the decoder, a genuinely different architecture from the 31B Dense, which is why 0.21.0 rejected it. The single-GPU 12B load test is the first task — the whole delegation architecture depends on the sub-agent tier actually serving.
- **The blocker is solved via a dedicated launch image, not a stable release** (per the vLLM recipe for `google/gemma-4-12B-it`, updated 2026-06-04). Support for the unified architecture landed in vLLM PR #44429 and has *not* shipped stable; the path is the pinned image `vllm/vllm-openai:gemma4-unified` (CUDA 13; `-cu129` tag for CUDA 12.9 hosts) or a nightly wheel. Same posture as the Week 8 Gemma-4 day-1 work — a launch build, not `pip install -U`. Holds the "upstream images only, no custom Dockerfiles" rule.
- **First task, do-or-die for the sub-agent tier: does `gemma4-unified` load the *w4a16 QAT* checkpoint on a 24 GB card?** The recipe documents the *BF16* 12B and says it needs a 40 GB+ GPU — which the x1-card 3090s (24 GB) do not have. That's exactly why the QAT was chosen. But the recipe only documents BF16; whether the unified image loads the 4-bit QAT variant is unstated. If yes → sub-agent tier is viable on this hardware. If no → there's a fit problem to solve (alternate quant, or the 12B doesn't fit the sub-agent cards at all), which would reshape the architecture. Test this before anything else.
- **Two vLLM builds, one per tier — not a single converged version (yet).** The `gemma4-unified` image is nightly-based; the 31B FP8 work is pinned to stable 0.21.0. There is no single release today that carries both the unified path and the 31B Dense FP8 path. So the near-term reality is the orchestrator on 0.21.0 and the sub-agents on `gemma4-unified`. This is fine for the Week 12 *load test* (it needs only the unified image). Single-version convergence — and the clean 31B re-baseline that depends on it — is a Week 13 / later concern for the concurrent deployment, gated on vLLM's release timeline, not a Week 12 blocker. Evaluate 0.22.x / K=V-unification as part of that convergence (it would also lift the ~2× global-layer over-allocation noted in Week 9's epilogue and Week 11 Day 6).
- **Recovered-util adoption decision:** whether the CUDA-graph-tax-recovered util (0.97, +22.7% usable context on TP=2) becomes the orchestrator's baseline. It changes held-constant, so it's a deliberate choice requiring re-baselining — and the two vLLM-recommended recovery recipes are non-viable on this 24 GiB hardware, so the working path (util 0.97, found by laddering) must be documented as the one that holds. Tied to the convergence step above (the re-baseline is where this gets locked in).
- **Box layout, now evidence-backed:** orchestrator = 31B TP=2 on the NVLink pair (GPUs 0+2); sub-agents = two independent 12B-QAT workers, one per PCIe-x1 card (GPUs 1 and 3). Two separate single-GPU instances, not tensor-parallel across the x1 link (Week 3/6 established that's non-viable on x1).
- **Capability bonuses for the delegation pattern** (from the recipe, worth noting for Week 13's orchestrator↔sub-agent wiring, not for the load test): native function calling with a custom tool-use protocol (`--tool-call-parser gemma4`) and structured thinking mode (`--reasoning-parser gemma4`) — tool-calling sub-agents are exactly what a fan-out orchestrator wants. An MTP assistant draft model exists as a later decode-speed lever. Context pins to 128K (`max_position_embeddings 131072`) though the card markets 256K — characterize the real ceiling rather than trusting either number.
- **Deliverable:** sub-agent tier characterization (12B-QAT single-GPU on `gemma4-unified`: does it load on 24 GB, context ceiling, decode/prefill on an x1 card), the QAT-on-unified-image compatibility finding, and a go/no-go on the delegation architecture. (The 31B re-baseline moves to the convergence step / Week 13.)
- **Models tested:** Gemma 4 12B-QAT (`google/gemma-4-12B-it-qat-w4a16-ct`) on `vllm/vllm-openai:gemma4-unified`.
- **Outcome (Week 12 complete):** Sub-agent tier **validated** — go on the delegation architecture. The QAT 
  checkpoint loads and serves on a single 24 GB card (8.28 GiB weights; Day 1's OOM was self-inflicted via a shallow-replacing `--hf-overrides`, plus one genuine image bug patched via a 3-line upstream backport — retire at version convergence). No memory ceiling exists on this card: the full 262,144 architectural context fits at 2.16× concurrency. **Production MML: 131,072** — the model's `max_position_embeddings` validation boundary; memory permits 262K but the 131K–262K range is quality-unvalidated (see the Week 12 summary journal for the full rationale and the long-context evaluation open item). Measured: decode 69.6 tok/s @8K / 51.7 @64K / 46.2 @102K; batching pays 2.33× at 8K but the worker is functionally serial at 64K+ — a direct input to Week 13's front-door design (queueing ≈ batching at depth, with better latency). The "characterize the real ceiling rather than trusting either number" instruction above is resolved: the card's 256K is mechanically real, the config's 128K is the validated envelope, and we ship the latter.

### Week 13: Two-tier QAT quality characterization (planned: concurrent two-tier serving) ✅

*Repurposed from the original "Quantized Model Serving with vLLM" block. This is the delegation architecture from Week 12 made real: 31B orchestrator and 2×12B sub-agents running concurrently as three independent services on the four-GPU box, fronted by a single endpoint. It also revives the nginx/reverse-proxy work deferred as a side-quest in Week 10 (`inference-reference-stack`) — now load-bearing rather than busywork, because the two-worker sub-agent tier gives the reverse proxy an actual job.*

- **Concurrent three-service deployment:** 31B TP=2 on the NVLink pair plus two single-GPU 12B-QAT workers on GPUs 1 and 3, all live at once. Process/port management, VRAM partitioning across all four cards, and ensuring the three vLLM instances coexist without contention on load.
- **nginx front door:** two separate 12B models behind one nginx endpoint, reviving the deferred Week 10 stack work. **Open design question:** load-balanced pool (nginx round-robins across the two workers — "give me any free worker") vs. path-addressable workers (the orchestrator targets a specific worker — "worker A handles component X"). The fan-out pattern of the orchestrator decides this; flag it, don't pre-decide.
- **Cross-tier interference characterization:** the three services share host RAM, the PCIe root complex, and CPU even though they're on separate GPUs. The honest systems question: does the orchestrator's latency stay isolated when the sub-agents are saturated, or do the tiers interfere? This is the measurement that validates the box layout works as a concurrent system, not just as three configs that each boot.
- **Version convergence + 31B re-baseline (moved here from Week 12):** the concurrent deployment is where the two-build split has to be resolved — ideally onto a single vLLM version carrying both the `gemma4_unified` path and the 31B Dense FP8 path. If no converged version exists yet, the fallback is running the two tiers on two different upstream images side by side (orchestrator on 0.21.0, sub-agents on `gemma4-unified`), which is operationally workable but worth documenting as a known split. When convergence happens, re-run the Week 11 TP=2 measurements as a regression check (Week 11's pinned-0.21.0 numbers are the baseline) and lock in the recovered-util decision.
- **Observability across both tiers:** extend the Week 10 Prometheus/Grafana stack to cover all three services — per-tier KV usage, latency, throughput — so interference is visible.
- **Deliverable:** a working two-tier orchestrator/sub-agent deployment with a single endpoint, the routing-approach decision documented, and concurrent-interference measurements. This is the operational proof that completes the delegation-architecture arc started in Week 11 — and, if it holds, the evidence behind the architecture Pulse held since Week 11.
- **Models tested:** Gemma 4 31B Dense FP8 (orchestrator) + 2× Gemma 4 12B-QAT (sub-agents), concurrent.
- **Outcome (Week 13 complete):** the week ran primarily as a **two-tier QAT quality-characterization** effort rather than the planned concurrent-serving/interference study. Version convergence landed first — both tiers moved onto a single pinned `vllm/vllm-openai:v0.23.0` image, retiring the Week-12 two-build split, the source-patched launcher, and the QAT load workarounds (the `patch_dense` fix is upstream; QAT loads clean with quantization genuinely active on the Marlin WNA16 kernel). On that converged stack, **QAT W4A16 was characterized against the BF16 parent at both tiers** (31B orchestrator Day 8, 12B workers Day 9) across both worker components (payment-service, order-service): quality-equivalent throughout — guardrail adherence an 8/8 tie at the orchestrator, pointwise 4.83–5.0 at the workers, format conformance 6/6. Built the evaluation toolchain (`rca_quality_judge.py` — position-bias-controlled pairwise + pointwise judge; `rca_quality_probe.py`; `worker_contract_check.py`; `vllm-bringup-checks.sh`) and ran the QAT-vs-FP8 throughput benchmark (QAT decode +36–50%). Published the LLM-as-judge quality-methodology LinkedIn Pulse. This **pulled a focused slice of the Week-24 quality work forward** — deployment-equivalence of QAT vs parent, distinct from the broad quant-fidelity curve that remains in Week 24.
- **Carried to Week 14:** the concurrent stack runs, but nginx load-balancing is broken (8/0 worker distribution, diagnosed not fixed), and the cross-tier interference characterization + the architecture write-up (the operational proof of the delegation architecture) remain — they become Week 14's nginx-fix and architecture-write-up items.
- **Headline finding:** QAT W4A16 is quality-equivalent to the BF16 parent at both tiers and production-ready.
- **Journals:** Day 8 (31B BF16-vs-QAT), Day 9 (12B worker-component quality, both components).

### Week 14: Phase 3 close-out — repo maintenance, operational proof & throughput characterization

*A loose-ends / consolidation week closing Phase 3: finish the delegation-architecture operational proof carried from Week 13, put the toolchain and repo in order, and characterize the worker tier's parallelism options. Cadence note: Week 14 switches session naming from `dayN` to `sessionN`.*

- **Repo maintenance (Session 1).** Split the toolchain — and the eval inputs (prompts, probes, rubrics), which move with it — into a new public repo `T` (`ai-training-tools`); the current repo stays public as `R` (results, journals, captures). Putting the inputs in T means T's SHA pins code + inputs together. Dissolves the results-dirty-the-tools provenance friction at the root — supersedes the auto-commit and path-classifier ideas explored in Week 13, both dropped. **Crux:** the capture / judge / check tools must record **T's** commit, not the working directory's (post-split that's R), or the problem relocates. Also **reorganize R** for consistency — the weekly reports are currently split between `docs/weekly-reports/` and `phase3-…/week-N/`; consolidate to one convention. Start the session with a full recursive listing of R to plan the reorg.
- **nginx load balancing.** The `zone workers 64k;` fix + balance re-probe (the diagnosed 8/0 worker distribution → balanced); nginx directory-mount so `reload` survives git's inode swap. This unblocks the **architecture write-up** — the delegation-architecture operational proof held since Week 11 — whose load-balance claim waits on a re-probe showing balanced distribution.
- **Throughput Pulse.** The QAT-vs-FP8 decode advantage (+36–50%) with the context-headroom hook — "everyone checks whether the weights fit; nobody checks how much context fits beside them." Tables built as ASCII inside fenced code blocks (Pulse does not render Markdown tables).
- **12B-QAT parallelism throughput sweep.** Characterize the 12B QAT worker at **TP=2, TP=1, and PP=2** — the worker-tier deployment decision (one card vs two, and TP vs PP on it), complementing the Week-11 31B parallelism characterization at the smaller tier.
- **Deliverable:** the toolchain in its own public repo with a reorganized results repo; balanced two-tier serving plus the architecture write-up completing the delegation-architecture arc; the throughput Pulse; and a 12B-QAT parallelism characterization.
- **Models tested:** Gemma 4 12B-QAT (parallelism sweep); the converged two-tier stack (nginx balance re-probe).

### Week 15: Operational Proof — Cross-Tier Interference Characterization & the Architecture Write-Up

*Redefined from "Speculative Decoding & KV Cache Compression" as the first of two concluding weeks. This is the one substantive measurement the program still owes itself: the proof that the two-tier box works as a **concurrent system**, not three configurations that each boot. The week runs on the frozen production configuration — 4× RTX 3090, pinned `vllm/vllm-openai:v0.23.0` — which is the exact state the Week 16 capstone will describe.*

- **Cross-tier interference characterization** (carried since Week 13): the three services share host RAM, the PCIe root complex, and CPU even though they occupy separate GPUs. Does orchestrator latency stay isolated while both 12B workers are saturated — and workers' while the orchestrator is under load? Regimes: per-tier isolated baselines, workers-saturated with orchestrator probe, orchestrator-loaded with worker probe, and full concurrent load through the nginx front door. Predictions committed per tier and regime before measurement, per methodology; prediction errors documented with mechanism.
- **Delegation-architecture write-up, finalized**: the operational-proof section of the write-up requires this week's interference data, so whatever the Week 14 close-out advanced (the load-balance re-probe and draft), the write-up completes here with the concurrency evidence folded in. This closes the arc opened by Week 11's "neither config serves the use case" finding.
- **Explicitly out of scope**: vLLM version changes, hardware changes, and any new optimization topics. (A vLLM 0.23→0.24 go/no-go was considered for this week and rejected — the engine upgrade consolidates with the hardware migration into the successor program's platform-revalidation prologue as one convergence event, regression-tested against the baselines this program freezes.)
- **Deliverable:** interference characterization with committed predictions and per-tier results; the completed delegation-architecture write-up.
- **Models tested:** the converged two-tier production stack — Gemma 4 31B-QAT orchestrator (TP=2, NVLink pair) + 2× Gemma 4 12B-QAT workers — on pinned v0.23.0.

### Week 16: Program Conclusion — Renames, Journal Consolidation, Capstone & Method Pulse

*Redefined from "NSight Profiling & Bottleneck Analysis" as the concluding week. The capstone is **declared, not built**: the two-tier delegation stack, validated for quality (Week 13) and concurrency (Week 15), is the program's capstone deliverable already in production form. The week is sequenced so every final document is written against final repo names and consolidated journals.*

1. **Repo renames (first, before any final document is drafted)**: the results repo → `inference-engineering-training`; the toolchain repo → `ai-training-tools`. The program's identity anchors to the role and the work, not the hardware that happened to run it. Mechanics: GitHub renames create permanent redirects for web links and git remotes, so published Pulses and the upstream vLLM issue reference keep resolving; update `git remote set-url` on both machines; grep all repos (including `inference-reference-stack`) for cross-references to the old names; never reuse the old names (reuse breaks the redirect). Historical journals keep whatever names they mention, per the never-rewrite rule — the redirects keep those references live. The rename is recorded in the capstone.
2. **Journal consolidation**: create weekly summary journals for every week that lacks one (the weekly-summary pattern was only adopted late in the program) and a phase summary journal for each of the three phases. These are **additive** documents — daily journals are never rewritten or renamed. The phase summaries become the intermediate representation the capstone draws from.
3. **Capstone summary**: a comprehensive program record written to the renamed results repo's `docs/` — the program's spine in one document, from Week 1's 5,000 tok/s transformers plateau to the quality-validated, concurrency-proven two-tier production architecture on converged v0.23.0. Written for the program's own record and for a future reader of the repo; too comprehensive for a Pulse by design.
4. **Method Pulse**: a public post about structured AI-assisted self-training *as a method* — the program discipline (predict-before-measure, one experiment at a time, honest null results, journals never rewritten) and the public paper trail as evidence — deliberately distinct from the capstone's technical content. No tabular data anticipated, so the ASCII-table constraint is moot for this post.
5. **Plan closure**: final Key Changes entries (recorded below) and the footer status update conclude this document as the program's record.

The Parallel Learning Streams (below) conclude with the program; the knowledge-map receives its final update as part of the capstone consolidation.

---

## Disposition of Phases 4–6 (Weeks 17–28)

*The program concludes at Week 16. The written Phases 4–6 predate the delegation architecture; by Week 14 their substance was largely achieved ahead of the paper plan, and the remainder divides into work that migrates to the successor program and topics deferred to a potential future module. The original section texts are preserved in this file's git history; the disposition is recorded here and in the Key Changes log.*

**Achieved in substance by the delegation-architecture arc:**
- **Week 19 (Multi-Model Routing & Orchestration)** — the two-tier delegation architecture *is* multi-model orchestration, running: 31B orchestrator + 2×12B workers behind one nginx endpoint, with the routing decision documented in Week 13. The application-side routing logic (query-complexity classification) migrates (below).
- **Week 23 (Full Observability Stack)** — delivered by the `inference-reference-stack` Prometheus/Grafana/dcgm work (Week 10) extended across both tiers in Week 13.
- **Weeks 25–28 (Capstone: "Enterprise Inference Platform")** — the two-tier production stack is the platform in substance: multi-model serving, observability, graceful degradation, single front door. Week 16 declares it as the capstone rather than rebuilding it to the original spec.
- **Week 20 (Production Hardening) — partial** — graceful degradation and overload behavior were measured in Weeks 4–5; health checking exists in the bring-up tooling. The remainder (circuit breakers, retry logic, failover) migrates.

**Migrated to the successor program (`ai-engineering-training`):**
- **Weeks 17–18 (RAG pipeline)** → the successor's knowledge-grounding phase.
- **Week 19 (application-side routing logic)** → the successor's agents/orchestration phase.
- **Week 20 (hardening remainder: failover, resilience)** → the successor's system-hardening phase.
- **Week 24 (latency-quality tradeoff framework, incl. the broad quantization quality-degradation curve)** → the successor's evaluation work. (Week 13 already pulled the focused QAT-vs-parent equivalence slice forward.)

**Deferred to a potential follow-on inference-engineering module:** *(recorded as topics of continued interest, not commitments)*
- Speculative decoding & KV cache compression (original Week 15 content).
- NSight profiling & kernel-level bottleneck analysis (original Week 16 content).
- **Week 21 (Infrastructure Cost Modeling)** and **Week 22 (Capacity Planning Framework)** — capacity fundamentals were partially covered in Weeks 1 and 4; the systematic frameworks remain wanted.

---

## What Follows

*This program's successor is `ai-engineering-training` — an AI-engineering program building toward an operator copilot for root-cause analysis over a distributed system, defined in its own plan document. Before its first phase, a **prologue outside the numbered phases** brings the platform to a revalidated state (for this hardware: GPU migration and a vLLM engine upgrade, smoke- and regression-tested against the baselines this program froze at v0.23.0/4×3090) and stands up the lab environment the new program trains against. The migrated topics above land in the successor's phases as noted.*

---

## Parallel Learning Streams

**Throughout all phases:**

- Read NVIDIA technical blogs and GTC talks (weekly, 1-2 hours)
- Write technical blog posts about learnings (bi-weekly)
- Review and update "AI Infrastructure Knowledge Map" (monthly)

*The streams conclude with the program at Week 16.*

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
| Week 10: Triton + framework comparison | Week 10: Inference-reference-stack (vLLM + nginx + Prometheus + Grafana). Days 1-2 completed; nginx/TLS/auth/metering deferred as side-quest | NGC Triton vLLM image bundled vLLM 0.15.1, incompatible with the Gemma 4 AWQ-INT4 work — pivot to building a public reference deployment repo with the architectural concerns above the engine as the learning target. After Days 1-2, the remaining work (nginx, TLS, API keys, metering) is real but well-documented elsewhere; returning opportunistically beats completing serially before Week 11 |
| Week 11: Quantization methods (AWQ, GPTQ) | Week 11: TP vs PP comparison on Gemma 4 31B Dense FP8 | The parallelism-strategy thread (Week 3 → 7 → 8 Day 2 → 9 Day 2 paused) now has its closing chapter accessible thanks to the HMA fix in vLLM 0.21.0. Quantization methods (originally Week 11) moves to Week 12 |
| Phase 3 starts at Week 9 | Phase 3 starts at Week 11 | 2-week shift accommodates the Gemma 4 work in Weeks 8-9 and the inference-stack work in Week 10 |
| Phase 3 = 4 weeks (originally Weeks 9-12) | Phase 3 = 5 weeks (Weeks 11-15) | One additional week to fit the parallelism-strategy closing chapter alongside the original quantization/profiling content |
| Week 11-12: Custom CUDA Kernels | Week 15: Speculative decoding + KV cache compression, Week 16: NSight profiling | More production-relevant than writing custom kernels from scratch (originally Weeks 14-15; shifted +1 by the Week-14 close-out) |
| Week 12: Quantization Methods (AWQ, GPTQ) — quality measurement | Week 12: Sub-agent tier validation & the delegation architecture | Quantization-as-deployment-baseline already established across Weeks 8/9/11 (AWQ on Ampere, HMA KV savings, FP8); operating principle is highest-fidelity-that-fits, not quant-vs-quant perf. Week 11's max-MML work showed no single config serves the interactive use case, motivating a tiered orchestrator/sub-agent architecture — Week 12 validates the cheap (12B) tier, gated on the `gemma4_unified` vLLM-version investigation |
| Week 13: Quantized Model Serving with vLLM | Week 13: The delegation architecture, operational — concurrent two-tier serving | Repointed to the concrete system: 31B orchestrator + 2×12B sub-agents running concurrently behind one nginx endpoint. Revives the nginx/reverse-proxy work deferred as a Week 10 side-quest, now load-bearing. The two-tier interference characterization is the operational proof of the architecture |
| (quantization quality measurement) | Folded into Week 24 (Latency-Quality Tradeoff Framework) | The one part of the original quantization weeks still wanted — qualitative fidelity degradation at lower bit-widths — belongs with the existing latency-quality framework (which already covered quantization's UX effect), removing a redundancy rather than creating a standalone week. Week 13 separately pulled a *focused* slice forward (QAT-vs-parent deployment-equivalence), distinct from this broad quant-fidelity curve |
| Phase 5: Separate PM track | Phase 5: Integrated operations + cost modeling | Cost modeling benefits from having all benchmark data in hand |
| 24-week program | **27-week program** | The four-day Gemma 4 arc, the Week 9 continuation, and the parallelism-strategy closing chapter in Week 11 extend the timeline by three weeks; later phases preserved at original length rather than compressed |
| Week 13: concurrent two-tier serving (operational proof) | Week 13: two-tier QAT quality characterization (BF16-vs-QAT, both tiers); operational-proof remainder (nginx balance + architecture write-up) carried to Week 14 | The converged-stack work landed, but the week's center of gravity became the quality question (is QAT lossless vs the parent?), pulling a focused slice of the Week-24 quality work forward; the interference/nginx proof carries forward rather than being dropped |
| Week 14: Speculative Decoding & KV Cache Compression | Week 14: Phase-3 close-out (repo split + reorg, nginx load-balancing, throughput Pulse, 12B-QAT TP/PP sweep) | Loose ends accumulated through Phase 3 — the toolchain/results repo commingling, the deferred nginx balance fix, the held throughput Pulse, and the un-characterized worker-tier parallelism — warrant a consolidation week before profiling |
| Phase 3 = 5 weeks (Weeks 11-15) | Phase 3 = 6 weeks (Weeks 11-16) | The close-out week (Week 14) adds one week; speculative decoding → Week 15, NSight → Week 16. Downstream phases (4–6) shift +1 |
| 27-week program | **28-week program** | The Phase-3 close-out week (Week 14) extends the timeline by one; later phases preserved at original length |
| Week 15: Speculative Decoding & KV Cache Compression | Week 15: Operational proof — cross-tier interference characterization + delegation-architecture write-up | The program concludes at 16 weeks (see below). The one measurement still owed is the interference characterization carried since Week 13; the architecture write-up's operational-proof section requires that data, so both close together. Speculative decoding / KV compression deferred to a potential follow-on inference module |
| Week 16: NSight Profiling & Bottleneck Analysis | Week 16: Program conclusion — repo renames, journal consolidation, capstone summary, method Pulse | Concluding consolidation replaces further optimization depth; the capstone is declared (the validated two-tier stack), not built to the original Phase-6 spec. NSight deferred to a potential follow-on inference module |
| Phases 4–6 (Weeks 17–28) | Dispositioned: achieved in substance / migrated to `ai-engineering-training` / deferred to a potential follow-on module | Written before the delegation architecture emerged. By Week 14 the two-tier stack had delivered the substance of routing (W19), observability (W23), and the capstone platform (W25–28). RAG, application-side routing, hardening remainder, and the quality-degradation curve migrate to the successor; cost modeling, capacity planning, and the optimization-depth topics defer |
| Repos named `rtx3090-*` | Results repo → `inference-engineering-training`; toolchain repo → `ai-training-tools` | Program identity anchored to role and work rather than the hardware that ran it; executed early in Week 16 so all final documents carry final names; GitHub redirects preserve published links |
| 28-week program | **16-week program** | The conclusion is declared with the capstone achieved in substance ahead of the paper plan. The successor program (`ai-engineering-training`), with a platform-revalidation + lab bring-up prologue outside its numbered phases, takes the migrated work |

---

*Training started: January 13, 2026*
*Current status: Weeks 1–13 complete; Week 14 (Phase-3 close-out) concluding. Weeks 15–16 defined above as the program's conclusion. Program finalized at 16 weeks (Phase 3 → Weeks 11–16; Phases 4–6 dispositioned — see Disposition section).*
*Hardware: 4x RTX 3090 (96GB total), Gigabyte B650 Eagle AX, Ubuntu 24.04*
*NVLink bridge: Installed (AORUS GeForce RTX NVLink, GPU0+GPU2, NV4)*
*Successor: `ai-engineering-training` — see its plan for the platform-revalidation and lab bring-up prologue.*
