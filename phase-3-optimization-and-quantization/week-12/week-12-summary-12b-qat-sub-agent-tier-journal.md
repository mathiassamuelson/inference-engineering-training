# Week 12 Summary — 12B QAT: from "won't load" to a production-configured sub-agent worker

**Dates:** 2026-06-09 → 2026-06-12 (Days 1–4)
**Model:** `google/gemma-4-12B-it-qat-w4a16-ct` — the 12B-parameter Gemma 4, quantization-aware-trained (QAT) to 4-bit weights with 16-bit activations (w4a16)
**Hardware:** single RTX 3090 (24 GB), GPU 1 — one of the PCIe 3.0 x1 cards
**Image:** `vllm/vllm-openai:gemma4-unified` @ `sha256:e828735f...63ed450` (pinned), plus a 3-line source patch (`patches/gemma4_unified.py`)

## TL;DR

The week opened with a do-or-die question: does the 4-bit 12B fit and serve on a single 24 GB card? Day 1 said no — out of memory. Day 2 proved the failure was self-inflicted (our own workaround had silently disabled quantization), patched a genuine image bug, and showed the model not only fits but supports its **full 262,144-token architectural context with 2.16× concurrency to spare**. Day 3 characterized throughput across context depths and concurrency levels and made the production call: **the sub-agent worker ships at max-model-len (MML) 131,072**. Three sessions took the model from "won't load" to a characterized, production-configured worker — and the deployment-tier comparison against the 31B orchestrator (below) confirms the two-tier box layout on both speed and memory grounds.

## The debugging arc (Days 1–2)

Day 1's load attempt produced three different crashes in sequence. The first two were stepped over with config workarounds; the third — an out-of-memory crash during weight loading, with ~22.85 GiB allocated and a 1.88 GiB `lm_head` allocation failing — looked like a genuine memory limit. Day 1's analysis blamed the two full-precision vocabulary tables (262K-word vocabulary, ~2 GiB each, exempt from 4-bit compression) plus CUDA-graph capture overhead.

Day 2 falsified the first repair hypothesis and found the real cause:

**Root cause #1 (self-inflicted): `--hf-overrides` does a shallow replace, not a merge.** Day 1's workaround for the second crash passed a partial `quantization_config` override containing only an `ignore` list. vLLM **replaced the checkpoint's entire quantization config with it** — no `quant_method`, no bit width — and silently fell back to building the model unquantized: every weight in bf16, ~24 GB for a 12B model. The card was never too small; Day 1's OOM was caused by Day 1's own workaround. The fix is to restate the checkpoint's *complete* `quantization_config` verbatim in the override, amending only the `ignore` list. The corrected override blob is embedded, with a warning comment, in the Week 12 launcher.

**Root cause #2 (genuine vLLM bug): missing `prefix` threading in the image's `gemma4_unified.py`.** The checkpoint ships `vision_embedder.patch_dense` un-quantized and lists it on the quantization skip-list — but the image constructs that layer without a `prefix` argument, so it reports an empty name, and a skip-list that matches by name can never match an empty name. The fix already existed in vLLM upstream main; we backported the 3-line change verbatim and mounted the patched file read-only over the image's copy. The patch only affects how the layer is *named* at construction, and the layer only executes for image inputs (disabled), so it cannot affect text-only measurements. **Retirement condition: Week 13 version convergence onto an image that postdates the upstream fix.**

With both fixed, the model loads clean: **8.28 GiB of weights** (less than the 9.56 GiB on-disk checkpoint — consistent with weight tying, where the output layer shares storage with the input embedding table), leaving an **~11 GiB KV-cache pool** on a 24 GB card at utilization 0.90.

## Context ceiling: there isn't one (Day 2)

Walking MML upward from 32,768 found no memory ceiling to find. The KV pool on this build is **MML-insensitive** (flat ~11.0 GiB across an 8× MML range — a contrast with vLLM 0.21.0, where the pool grows with MML; engine-version-dependent behavior). At the model's full architectural context of 262,144 tokens, the pool holds **2.16× concurrency** — matching the two-coefficient sliding-window cost model's prediction of 2.1–2.2×.

The mechanism, confirmed again on this model: Gemma 4 alternates sliding-window attention (SWA) layers, which keep KV only for a fixed recent window, with global layers, which keep KV for the entire sequence. Average per-token KV cost therefore *falls* with sequence length. **Day 4 resolved the exact layer split from the checkpoint config's `layer_types` field: 48 layers — 40 SWA, 8 global — a 5:1 ratio matching the 31B's 50/10 pattern.**

Consequences: the fp8-KV lever and the PP=2 fallback are both unneeded, and the recipe's 131,072 pin is **not** a memory constraint on this hardware — most likely a quality-validation boundary.

## Throughput characterization and the MML decision (Day 3)

All measurements on GPU 1, single card, CUDA graphs on, util 0.90, `tools/throughput_sweep.py` schema v3 (unique nonces, server-side token counts).

**Depth curve (c=1):**

| Context depth | Prefill tok/s | TTFT | Decode tok/s |
|---:|---:|---:|---:|
| 8K | 2,480 | ~3.3 s | 69.6 |
| 64K | 1,382 | ~47 s | 51.7 |
| 102K | 1,083 | ~94 s | 46.2 |

Smooth degradation, no cliffs. Prediction-vs-outcome attribution from Day 3 stands, sharpened by the Day 4 layer-count resolution: 8 of 48 layers carrying full-context attention is enough to dominate deep-context prefill cost — architecture intuition under-weighted the global layers, while extrapolation from two empirical points landed within range.

**Concurrency:**

- **@8K:** aggregate throughput scales 48 → 112 tok/s from c=1 to c=8 (2.33×); the regime is prefill-bound.
- **@64K:** aggregate 9 → 10.3 tok/s from c=1 to c=4 (1.15×) — **functionally serial.**

The serialization finding is the week's headline systems result: **at deep contexts, batching buys almost nothing on this worker.** Front-door queueing achieves roughly the same aggregate throughput as batching, with far better per-request latency. This goes directly into Week 13's front-door design.

**Functional verification at depth:** a 104K-token retrieval probe — a unique fact planted in the opening sentence of the context, ~104K tokens of synthetic filler, then a question asking for it back — **passed**, with the exact fact retrieved and coherent output. This is maximum-*distance* retrieval (only the 8 global layers can span it), and it verifies the plumbing: the configuration is functional end-to-end at depth. But it is easy on every other retrieval axis — a lexically unique needle, no distractors, a strongly-cued question, and a document-start placement that long-context models empirically favor ("lost in the middle": retrieval is strongest at the edges of the context, weakest in the middle). It says little about aggregation, paraphrased retrieval, or distractor robustness over real content. The long-context quality evaluation carried as an open item should plant needles at varied depths, especially mid-context.

**Operational findings:** the server enforces MML at admission with a clean HTTP 400, usable as a front-door routing signal; worker on GPU N serves host port 8000+N.

**The production call: ship at MML 131,072.** Memory permits 262K, but 131,072 matches the model's `max_position_embeddings` pin (the boundary the model was presumably validated to), comfortably covers the deep-context workload the tier exists for, and leaves the KV pool with substantial concurrency headroom for the shallow-context fan-out work where batching actually pays. The 131K-to-262K range remains available but unvalidated for quality; revisit only with a long-context quality evaluation in hand.

## Deployment-tier comparison: 31B orchestrator vs 12B sub-agent

A desk exercise from committed results — Week 11's 31B TP=2 measurements vs Day 3's 12B single-card numbers. **This is a deployment-tier comparison, not a controlled architecture experiment.** The configs differ in parallelism topology (TP=2 vs single GPU), quantization format (FP8-block vs w4a16 QAT), vLLM build (0.21.0 stable vs `gemma4-unified` nightly preview), memory utilization (0.95 vs 0.90), and the Week 11 numbers predate any CUDA-graph-tax handling. The deep-context rows are additionally depth-mismatched (31B at 49K vs 12B at 64K — in the 12B's *disfavor*, so the gap shown is conservative).

| Axis | 31B FP8 (orchestrator tier) | 12B QAT (sub-agent tier) |
|---|---|---|
| Topology | 2× RTX 3090 + NVLink (GPUs 0+2), TP=2 | 1× RTX 3090, any slot — PCIe x1 irrelevant (no inter-GPU traffic) |
| Weights | 15.85 GiB/GPU (~31.7 GiB total) | 8.28 GiB |
| Decode @8K, c=1 | 40.2 tok/s | 69.6 tok/s (~1.7×) |
| Decode, deep context | 33.4 tok/s @49K (probing its 54,496 ceiling) | 51.7 tok/s @64K (~1.5×, at greater depth) |
| Prefill @8K, c=1 | 1,689 tok/s | 2,480 tok/s (~1.5×) |
| Prefill, deep context | ~1,130 tok/s @49K | 1,382 tok/s @64K |
| TTFT @8K | ~4.8 s | ~3.3 s |
| Context ceiling | 54,496 MML (util 0.95; KV-starved — ~17 of 24 GiB/GPU is weights) | 131,072 shipped; pool holds full 262,144 at 2.16× |
| Batching headroom | ~1.25× max concurrency at MML 33024 — effectively single-request | 2.33× aggregate scaling @8K (c=8); serial at 64K+ |
| Quantization | FP8-block | w4a16 QAT (compressed-tensors) |
| Engine | vLLM 0.21.0 (pinned stable) | `gemma4-unified` nightly + 3-line source patch |

What the table says about the box layout:

1. **Routing is forced by memory before speed even enters.** Anything over ~54K of context *cannot* be served by the 31B — deep-context jobs route to the 12B tier on capacity grounds alone. Speed preference and memory necessity point the same direction.
2. **The 12B's indifference to PCIe x1 is what makes GPUs 1 and 3 productive at all.** A single-GPU deployment has no inter-GPU traffic, so the x1 link never enters the critical path — the slots Week 11 wrote off for every parallel topology are exactly right for single-card workers.
3. **The per-request speed advantage narrows at depth (1.7× → ~1.5× decode).** Both models pay the global-layer attention tax; the 12B's 8 global layers over 100K+ tokens cost proportionally more than its size advantage suggests.
4. **Both tiers go serial at depth, at different depths.** The 31B is effectively single-request at its working MML already; the 12B holds 2.33× batching at 8K but collapses to serial by 64K. Fan-out gains come from shallow-context parallelism on the workers — not from deep-context batching anywhere on the box.

## Methodology lessons logged this week

- **A partial `--hf-overrides` is a loaded gun:** it shallow-replaces entire config objects. Restate them verbatim or don't touch them. (Day 1's OOM.)
- **Read the engine's config echo line** — `quantization=None` was sitting in the failed boot's log the whole time.
- **Falsify cheap hypotheses by order-of-operations reasoning first:** compile/graph capture happens *after* weight load, so `--enforce-eager` could never have fixed a load-time OOM.
- **Once two empirical points exist, extrapolate from them, not from architecture intuition.** The theory-based 64K predictions missed; the trend-based 102K predictions landed.
- **Chat-template-vs-raw-endpoint degeneracy has a recognizable signature** — probe the endpoint you intend to measure.
- **Char-based prompt sizing is ±5–10%; trust server token counts only.**
- **Pull config facts from the config, not recollection** — the "~30 layers" memory was off by a wide margin (actual: 48, with 40/8 SWA/global).

## Artifacts produced this week

- `patches/gemma4_unified.py` (+ `.orig`) — 3-line upstream backport with provenance in the commit message; **retire at Week 13 version convergence**
- `tools/start-12b-qat.sh` — temporary Week 12 launcher (patch mount + full override blob); same retirement condition
- `results/day2-context-ceiling-walk_gemma-4-12B-it-qat-w4a16-ct.json` — launches, rungs, probes, cost-model fit
- Seven Day 3 sweep JSONs — `results/throughput_sweep_vllm-openai_gemma-4-12B-it-qat-w4a16-ct_c*_20260612T*.json`
- Per-day journals (Days 1–3; Day 4 was a desk session, recorded here)

## Carried into Week 13 (planning only — not started)

1. **Second worker on GPU 3 + nginx front door + cross-tier interference characterization.** Day 3's serialization finding shapes the front-door design: queueing ≈ batching for deep-context aggregate throughput, with better latency — the front door can be simpler than a batching-aware router.
2. **vLLM version convergence:** find one image serving both `gemma4_unified` and the 31B Dense FP8 path; retire the source patch and the temporary launcher; re-baseline the 31B against Week 11's numbers.
3. **Pending `start-vllm.sh` cudagraph-tax flag** (flag-gated opt-in) belongs to the upgrade work.
4. **131K-to-262K headroom** remains available pending long-context quality evaluation.
