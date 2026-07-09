# Phase 2 Summary — Production Inference at Scale (Weeks 5–10)

**Span:** February 2026 → April 2026 (Week 5: Feb; Week 6: Feb; Week 7: March; Week 8: Apr 3–7; Week 9: Apr 10–12 with a Day 4 re-test on May 17; Week 10: Apr 13 onward)
**Hardware arc:** 4× RTX 3090 throughout; the **AORUS NVLink bridge** arrived at Week 7 and landed on GPUs 0↔2 (NV4, ~100 GB/s) — the single most consequential hardware change of the program
**Model arc:** Llama 3.2 3B / Mistral 7B / Qwen 2.5 14B (Weeks 5–7) → the Gemma 4 family from its release day (Week 8 onward), which became the program's production line
**Frameworks:** vLLM 0.13.0 → 0.18.x/0.19.x day-1 builds → 0.21.0 (HMA); llama.cpp; Triton Inference Server (introduced Week 6, never became the serving layer — see below)
**Sources:** `week-05.md`, `week-06.md`, `week-07.md`, the Week 8/9/10 daily journals, and the Week 8/9/10 summary journals derived from them in Week 16

## TL;DR

Phase 2 turned the Phase 1 baselines into production serving knowledge in two distinct movements. **Weeks 5–7 were the framework-and-interconnect movement:** vLLM data parallelism at 95.4% efficiency across 4 GPUs (18,053 tok/s) and a measured **7.12× system-throughput advantage over transformers** on identical hardware; then a 14B tensor-parallel experiment whose grim economics (316.5 tok/s peak, ~20× cost per token) turned out to be a measurement of PCIe x1, not of 14B models — the NVLink bridge's arrival in Week 7 lifted the same configuration **9.53×** to 3,018 tok/s and rewrote the conclusion. **Weeks 8–10 were the frontier movement:** a deliberate pivot to day-1 deployment of the just-released Gemma 4 family produced a fixed upstream segfault, a corrected benchmark methodology (server-side prefix caching silently corrupting prefill numbers — twice, in two engines), the motherboard P2P topology finding, a dense-vs-MoE characterization, the program's first shared benchmarking tool, and a quantified contribution to a real vLLM KV-sizing bug (#39133) — followed by the discipline call that defined the program's character: **pause rather than publish on top of a known bug.** Week 10 closed the phase by building the deployment scaffold (`inference-reference-stack`) and the observability layer the later phases ran on. The phase's central lesson, learned three different ways: *the substrate you measure through (framework, interconnect, cache behavior) is part of the measurement, and must be characterized before conclusions are drawn.*

## Weeks 5–7 — framework advantage, interconnect truth

**Week 5 (measured):** vLLM data-parallel across 4 GPUs scaled at 95.4% efficiency to 18,053 tok/s (concurrency 256), beating Week 3's manual data parallelism on both efficiency and absolute numbers. Sustained-load testing showed rock-solid stability (CV 0.023 over 60 s) and continuous batching's fairness under mixed traffic (88.6–90.8 tok/s per request regardless of output length; 0.36 s quick replies coexisting with 4.8 s long generations). The head-to-head that settled the framework question: **transformers + multiprocessing 260 tok/s vs vLLM 1,852 tok/s on identical hardware and workload — 7.12×, entirely from architectural batching** (per-request latency was near-identical at 1.04×). Mistral 7B came in at almost exactly half the 3B's throughput with more graceful degradation — the arithmetic-intensity argument, observed.

**Week 6 (measured, later superseded in part):** Qwen 2.5 14B under TP=2 across GPUs 0+1 produced 38.6 tok/s single-request and a hard plateau at 316.5 tok/s — per-GPU efficiency ~20× worse than the 7B on one card, cost per token 19.69× higher. The week's decision framework ("smallest model that meets quality; larger models only for quality-critical paths") was built on those numbers. Week 6 also deployed Triton Inference Server with an ONNX embedding model — dynamic batching delivering 3.5× (23.2 req/s), Prometheus metrics, 345 MB footprint — the program's first and, as it turned out, only Triton deployment.

**Week 7 (the in-phase supersession, measured):** the NVLink bridge arrived — and topology verification found it on **GPUs 0↔2, not 0↔1 as assumed**, a small correction with a long shadow (the verify-placement-empirically habit starts here). The same 14B TP=2 configuration on the NVLink pair: **3,018 tok/s peak, 9.53× over the PCIe result**, with near-jitter-free latency (CV 0.6%, p50→p99 spread 35 ms over 30 trials). The week's own framing of the correction is the phase's methodological centerpiece: *"The Week 6 finding was not a conclusion about 14B models. It was a conclusion about 14B models over PCIe x1."* The bottleneck moved from communication to compute — the correct bottleneck, because compute saturation responds to quantization and scaling while interconnect saturation responds to nothing but new hardware. The output-length sweep added the prefill-amortization curve (2,280 tok/s at 25-token outputs → 3,018 at 200) to the capacity-planning toolkit.

## Weeks 8–9 — the Gemma 4 pivot: day-1 deployment as curriculum

Week 8 abandoned the planned Triton deep-dive to deploy Google's Gemma 4 within 24 hours of release — a deliberate bet that frontier-deployment reality would teach more than the scheduled material. It paid:

- **The 31B Dense arc:** deployed on llama.cpp release day +1; a prompt-length segfault binary-searched to its boundary, reported upstream, fixed overnight, verified. Corrected 31B baseline: ~1,170 tok/s prefill plateau, 20–24 tok/s decode.
- **The benchmark-artifact correction (Week 8), then its exact twin (Week 9):** server-side prefix caching silently converted "prefill throughput" into cache-lookup speed — first in llama.cpp's slot cache, then in vLLM's always-on prefix cache (71.2% hit rate producing apparent 29,000 tok/s prefill). Both caught by cross-checking against server-side counters; both fixed with per-request nonces. Learning the same lesson in two engines is what made it a *method* (every benchmark needs a server-side cross-check it didn't compute) rather than a bug fix.
- **The topology finding:** every non-NVLink GPU pair on this motherboard reports P2P unsupported — CUDA silently stages through host memory. This sharpened Week 7 in hindsight: NVLink's value was not just bandwidth but *peer access existing at all*.
- **Dense vs MoE (Week 8):** the 26B-A4B MoE decoded 4.6–4.7× faster than the 31B Dense at every context length and fit the full 262K window where the dense model auto-shrunk to 104K — per-cell KV cost (layers × KV heads), not parameter count, is the memory lever.
- **The six-failure vLLM bring-up (Week 8 Day 4):** the FP8 path on Ampere dead-ends (FP8 *weights* emulate anywhere via Marlin; FP8 *KV cache* needs SM 8.9+ — two different hardware requirements masquerading as one feature); AWQ-INT4 via compressed-tensors worked and served the full 262K window at TP=2.
- **Week 9:** built `tools/throughput_sweep.py` (self-describing output, provenance metadata — the seed of the eventual T-repo toolchain), reframed "vLLM vs llama.cpp" into the better-posed "TP=2 vs layer-split," measured two clean single-request crossovers (decode ~8K, prefill ~32K) — and then found that vLLM's KV allocator was ignoring Gemma 4's hybrid attention entirely (all 30 layers sized as full-context; observed bytes/token matched that hypothesis to 0.04%). Contributed the quantified reproduction to upstream issue #39133 — and **paused the week** rather than publish crossover conclusions or run concurrent benchmarks on a known-buggy allocator. The Day 4 re-test (May 17, vLLM 0.21.0's Hybrid Memory Allocator): **9.3× KV pool capacity** from the same VRAM, single-request throughput unchanged — the bug was allocator bookkeeping, not attention math, so what it had been suppressing was concurrency, not speed.

## Week 10 — the deployment scaffold

The phase closed by building where the program would live: the public **`inference-reference-stack`** repo — Docker Compose with vLLM, Prometheus, Grafana, DCGM-exporter; streaming-aware nginx config declared (launched later, in Phase 3); digest-pinned engine images as the reproducibility anchor for the pending #39133 re-test. The planned Triton serving layer was dropped on a concrete version conflict (the NGC image bundled a vLLM three minor versions too old for the Gemma 4 quant path) with the decision documented in-repo: the stack's architectural value lives above the engine. Two dashboard-era lessons stuck: vLLM metric names drift across versions (dashboards are implicitly digest-pinned), and per-GPU panels double as a live topology check. No measurement was performed in Week 10; it was pure infrastructure.

## Supersessions and corrections across the phase

- **Week 6's 14B economics → rewritten by Week 7.** The 19.69× cost penalty and "not a universal upgrade" framing were substrate artifacts; NVLink made the 14B a viable production target. The Week 6 report stands as the record of what PCIe x1 does to tensor parallelism.
- **Phase 1's "tensor parallelism unusable on this hardware" → conditionally repealed.** True for the topology Phase 1 measured; the NVLink pair changed the premise. The Week 8 P2P finding then explained *why* the non-bridged pairs were so much worse than bandwidth math predicted.
- **"NVLink bridge on GPUs 0+1" (assumed) → GPUs 0+2 (verified).** The first instance of what became a standing rule: empirical placement over intent, every time.
- **Week 9's TP-vs-layer-split crossovers → declared suspect the same week** once #39133 surfaced, and left un-readjudicated even after the Day 4 re-test showed single-request throughput unchanged — the question was parked pending concurrent measurement rather than quietly resolved.
- **The Triton thread → introduced (W6), displaced (W8 pivot), displaced again (W9 → W10 plan), dropped on version conflict (W10).** The deep-dive never happened; the honest record is that the Gemma 4 work and the standalone-vLLM stack were judged more valuable each time the choice arose.
- **Week 1's ~5,000 tok/s plateau → contextualized:** Week 5's cross-week table shows the same hardware at 18,053 tok/s under vLLM — the plateau was the framework, as Phase 1 hypothesized and Phase 2 proved.

## Not measured in this phase (as the record left it)

- **Concurrent-load benchmarking on the fixed (post-HMA) build** — deferred at Week 9 close; the engine-reported concurrency figures scattered through Weeks 8–9 are capacity math, never load tests. (Picked up in Phase 3.)
- **True vLLM prefill rates** — all TTFT-based, lower bounds by an unknown margin (llama.cpp's server-side counter showed its own TTFT-based numbers 20–33% low; vLLM has no equivalent signal).
- **The Triton-vs-vLLM LLM-serving comparison** planned at Week 6 — never ran.
- Week 10 shipped configuration only; its stack was first load-exercised in Phase 3.

## What Phase 2 handed Phase 3

- **The NVLink-pair / x1-worker geometry:** TP belongs on GPUs 0+2; the x1 slots are for single-card work — the box layout Phase 3's two-tier architecture was built on.
- **The Gemma 4 family as the production line**, with its hybrid-attention KV economics (5:1 SWA/global ratio, per-cell cost levers) characterized from both engines' vantage points.
- **The toolchain seed:** `throughput_sweep.py` with nonce discipline, tokenizer calibration, provenance metadata, self-describing filenames — the conventions that became the T repo's rules.
- **The IRS stack:** Compose, observability, digest pinning, the declared-but-unlaunched nginx front door.
- **The method, now explicit:** predict before measuring; cross-check against a counter you didn't compute; verify placement empirically; search upstream before investigating; and don't publish on top of a known bug — pause and re-test.
- **An open upstream thread:** #39133's remaining K=V-unification gap and the ~400 MiB fixed overhead, identified but not root-caused.

## Register notes

- Weeks 5–7 report month-level dates only; Week 8 onward carries per-day dates. Week 5–7 reports describe the program as "Week N of 24" — accurate to the plan of their moment (the program was later finalized at 16 weeks; the plan's Key Changes log records the evolution).
- All throughput/latency tables above are **measured**; cost-per-token and users-per-SLA tables in Weeks 5–6 are **period planning estimates** built on those measurements, and the Week 6 cost table is additionally **superseded** by Week 7's interconnect correction.
- Week 5's GPU-1 ~4% deficit ("likely thermal positioning") and Week 6's saturation attributions are **interpreted-as-recorded**; neither was isolated experimentally.
- The Week 8–9 material here compresses the fuller weekly summaries (`week-08-summary-…`, `week-09-summary-…`), which carry the per-claim register detail (prediction misses, noise bands, confound inventory); this phase document defers to them.
- Weeks 8–10 predate the R/T repo split and the repo renames; historical references above use the names and layouts of their time where the distinction matters.
