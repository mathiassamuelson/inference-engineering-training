# Training Plan: NVIDIA Inference Stack Mastery (Revised)

*Last updated: June 22, 2026 — Weeks 12 and 13 complete. Week 13 ran as a two-tier QAT quality-characterization week (BF16-vs-QAT equivalence at both tiers via a position-bias-controlled LLM-as-judge harness) on a converged `vllm/vllm-openai:v0.23.0` image, rather than the originally-planned concurrent-serving/interference focus; the operational-proof remainder (nginx load-balancing fix + architecture write-up) carries into Week 14. Week 14 is repurposed from "Speculative Decoding" to a Phase-3 close-out / loose-ends week (repo split + reorg, nginx load-balancing, the throughput Pulse, and a 12B-QAT TP/PP throughput sweep). Speculative decoding shifts to Week 15 and NSight profiling to Week 16, extending Phase 3 to Weeks 11–16 and the program to 28 weeks (downstream phases shift +1). Quality-degradation measurement remains folded into the latency-quality week (now Week 24).*

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

## Phase 3: Optimization & Quantization (Weeks 11-16)

*Goal: Master parallelism strategies, then validate and operationalize the tiered delegation architecture that emerged from the Week 11 parallelism work. Phase 3 starts at Week 11 and runs 6 weeks. Weeks 12–13 were repurposed from the original quantization-methods content (rendered largely redundant by the Gemma deployment work in Weeks 8/9/11) to sub-agent-tier validation and two-tier QAT quality characterization; Week 14 is a Phase-3 close-out / loose-ends week. The originally-planned speculative-decoding and NSight content shifts to Weeks 15–16. The surviving quality-degradation measurement moved to Week 24.*

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

- **Repo maintenance (Session 1).** Split the toolchain — and the eval inputs (prompts, probes, rubrics), which move with it — into a new public repo `T` (`rtx3090-ai-training-tools`); the current repo stays public as `R` (results, journals, captures). Putting the inputs in T means T's SHA pins code + inputs together. Dissolves the results-dirty-the-tools provenance friction at the root — supersedes the auto-commit and path-classifier ideas explored in Week 13, both dropped. **Crux:** the capture / judge / check tools must record **T's** commit, not the working directory's (post-split that's R), or the problem relocates. Also **reorganize R** for consistency — the weekly reports are currently split between `docs/weekly-reports/` and `phase3-…/week-N/`; consolidate to one convention. Start the session with a full recursive listing of R to plan the reorg.
- **nginx load balancing.** The `zone workers 64k;` fix + balance re-probe (the diagnosed 8/0 worker distribution → balanced); nginx directory-mount so `reload` survives git's inode swap. This unblocks the **architecture write-up** — the delegation-architecture operational proof held since Week 11 — whose load-balance claim waits on a re-probe showing balanced distribution.
- **Throughput Pulse.** The QAT-vs-FP8 decode advantage (+36–50%) with the context-headroom hook — "everyone checks whether the weights fit; nobody checks how much context fits beside them." Tables built as ASCII inside fenced code blocks (Pulse does not render Markdown tables).
- **12B-QAT parallelism throughput sweep.** Characterize the 12B QAT worker at **TP=2, TP=1, and PP=2** — the worker-tier deployment decision (one card vs two, and TP vs PP on it), complementing the Week-11 31B parallelism characterization at the smaller tier.
- **Deliverable:** the toolchain in its own public repo with a reorganized results repo; balanced two-tier serving plus the architecture write-up completing the delegation-architecture arc; the throughput Pulse; and a 12B-QAT parallelism characterization.
- **Models tested:** Gemma 4 12B-QAT (parallelism sweep); the converged two-tier stack (nginx balance re-probe).

### Week 15: Speculative Decoding & KV Cache Compression

*Shifted from Week 14 when Week 14 was repurposed to the Phase-3 close-out.*

- Implement speculative decoding with a draft model (Llama 3.2 1B → 3B)
- Measure acceptance rate, latency improvement, and throughput impact
- KV cache compression techniques: sliding window, prefix caching
- **Deliverable:** Speculative decoding analysis with acceptance rate breakdown

### Week 16: NSight Profiling & Bottleneck Analysis

*Shifted from Week 15 when Phase 3 expanded to six weeks.*

- Profile inference workloads with NVIDIA NSight Systems and NSight Compute
- Identify compute vs memory bandwidth bottlenecks at the kernel level
- Profile vLLM's PagedAttention and continuous batching kernel execution
- **Deliverable:** Profiling report identifying top optimization opportunities

---

## Phase 4: Production Systems (Weeks 17-20)

*Goal: Build complete production-grade AI systems*

### Week 17: RAG Pipeline — Retrieval Infrastructure

- Build GPU-accelerated semantic search with FAISS
- Embedding pipeline: batch encoding, index construction, similarity search
- Benchmark retrieval latency vs index size
- **Deliverable:** Working vector search system with throughput benchmarks

### Week 18: RAG Pipeline — Generation & Evaluation

- Integrate retrieval with vLLM generation end-to-end
- Measure end-to-end latency: retrieval + generation
- Evaluate answer quality with and without retrieval context
- **Deliverable:** Complete RAG system with latency and quality measurements

### Week 19: Multi-Model Routing & Orchestration

- Build request router: classify query complexity, route to the larger vs smaller model (currently the Gemma 4 31B orchestrator and 12B worker; the specific pair may change by then)
- Measure routing accuracy and latency overhead
- Cost modeling: savings from routing vs single large model
- **Deliverable:** Adaptive routing system with cost analysis

### Week 20: Production Hardening

- Request queuing, timeout handling, graceful degradation under overload
- Health checks, circuit breakers, retry logic
- Load testing: find failure modes and recovery behavior
- **Deliverable:** Production-hardened inference service with reliability playbook

---

## Phase 5: Operations & Cost Modeling (Weeks 21-24)

*Goal: Build production operations and cost modeling capabilities*

### Week 21: Infrastructure Cost Modeling

- Build TCO calculator for this hardware configuration
- Compare on-premise (4x RTX 3090) vs cloud alternatives (Lambda Labs, RunPod, AWS)
- Break-even analysis across usage patterns and model sizes
- **Deliverable:** Cost modeling framework with measured data from this hardware

### Week 22: Capacity Planning Framework

- Model throughput, memory, and latency constraints together
- Capacity planning under uncertainty: traffic spikes, model updates
- SLA definition and measurement: p99 latency budgets, error rate targets
- **Deliverable:** Capacity planning guide for LLM serving systems

### Week 23: Full Observability Stack

- Deploy Prometheus + Grafana dashboards for all active serving infrastructure
- Track SLA-relevant metrics (p50/p95/p99 latency, throughput, error rates)
- Simulate failure scenarios and recovery
- **Deliverable:** Production monitoring dashboard and reliability playbook

### Week 24: Latency-Quality Tradeoff Framework (incl. quantization quality degradation)

- Document how quantization, batching, caching, and model size affect user experience
- **Quantization quality degradation** (folded in from the original Week 12): systematic measurement of how lower-bit models degrade — perplexity, generation coherence, task accuracy — across AWQ/GPTQ/FP8 at INT4/INT8 vs higher-fidelity baselines. This is the one piece of the original quantization weeks worth keeping: not deployment-performance (already covered by the Gemma work in Weeks 8/9/11), but the *qualitative* fidelity cost of going to fewer bits, which the operating principle ("run the highest fidelity that fits") makes a deliberate, measured tradeoff rather than a default
- Create decision framework for model selection (when to use 3B vs 7B vs 14B vs 70B, and at what quantization)
- **Deliverable:** Model selection guide with measured data from this hardware, including a quality-vs-bits curve

---

## Phase 6: Capstone & Portfolio (Weeks 25-28)

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

---

*Training started: January 13, 2026*
*Current status: Phase 2 complete (Week 10 partial — side-quest); Week 11 complete (TP-vs-PP + max-MML); Week 12 complete (sub-agent tier validated, 12B QAT shipped at MML 131,072); Week 13 complete (two-tier QAT quality characterization — QAT W4A16 ≡ BF16 parent at both tiers, production-ready); beginning Week 14 (Phase-3 close-out — repo maintenance, nginx balance, throughput, 12B-QAT parallelism sweep). Program now 28 weeks (Phase 3 → Weeks 11–16).*
*Hardware: 4x RTX 3090 (96GB total), Gigabyte B650 Eagle AX, Ubuntu 24.04*
*NVLink bridge: Installed (AORUS GeForce RTX NVLink, GPU0+GPU2, NV4)*
