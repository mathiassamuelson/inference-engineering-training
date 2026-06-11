# Week 12, Day 2 — 12B QAT: single-GPU load solved, context ceiling characterized

**Date:** 2026-06-11
**Model:** `google/gemma-4-12B-it-qat-w4a16-ct`
**Hardware:** single RTX 3090 (GPU 1, 24 GB), one of the PCIe-x1 cards — irrelevant here, since a single-GPU deployment has no inter-GPU traffic
**Image:** `vllm/vllm-openai:gemma4-unified` @ `sha256:e828735f...63ed450` (pinned), **plus a 3-line source patch** (see below)

## TL;DR

The model loads on one 24 GB card — Day 1's out-of-memory failure was **self-inflicted**, caused by our own Day 1 workaround silently disabling quantization. With that fixed (and a genuine image bug patched), the model occupies **8.28 GiB** of weights and leaves an **11 GiB KV-cache pool**. Walking the context limit upward showed the pool supports the model's **full 262,144-token architectural context with 2.16× concurrency to spare** — there is no memory ceiling to find on this card. Warmed decode rate: **~78 tok/s** with CUDA graphs (~69 tok/s in eager mode).

## The debugging arc: three launches, three different failures

### Launch 1: eager mode — same OOM, hypothesis falsified

Day 2 opened by testing the Day 1 theory that the out-of-memory crash came from torch.compile / CUDA-graph working memory building up during weight loading. We relaunched with `--enforce-eager` (which disables both). Result: **identical OOM at the identical spot** — 22.85 GiB allocated, died requesting 1.88 GiB for the `lm_head` (the model's final output layer). The theory was wrong for a simple order-of-operations reason: compilation and graph capture happen *after* weights load, so disabling them can't help a crash that occurs *during* loading.

But the failed run exposed the real culprit in its config line: `quantization=None`. **vLLM was building the model unquantized** — every weight in bf16 (16-bit floating point), ~24 GB for a 12B-parameter model. The expected 4-bit packed footprint was ~9.5 GB. The card was never too small; the quantization simply wasn't being applied.

### Root cause #1 (self-inflicted): `--hf-overrides` does a shallow replace, not a merge

Day 1's workaround for a load failure passed `--hf-overrides '{"quantization_config": {"ignore": [...]}}'`. We assumed this would *merge* into the checkpoint's quantization config. It **replaces the entire object**. The replacement contained only an `ignore` list — no `quant_method`, no bit width, nothing — so vLLM saw no usable quantization config and fell back to full-precision construction. Day 1's OOM was caused by Day 1's own workaround.

**Fix:** restate the checkpoint's *complete* `quantization_config` (read from its `config.json`) in the override, amending only the `ignore` list. This is now embedded, with a warning comment, in the Week 12 launcher script.

### Launch 2: quantization restored — original image bug now visible in its true form

With the full config in place, the model constructed in packed form and weight loading began — then failed on a shape mismatch: the checkpoint ships `vision_embedder.patch_dense` as a plain bf16 layer (it's on the checkpoint's quantization skip-list), but vLLM had constructed it as a packed/quantized layer. The skip-list wasn't being honored for this layer.

### Root cause #2 (genuine vLLM bug): missing `prefix` threading in `gemma4_unified.py`

Reading the model code inside the image showed why: the layer is constructed **without a `prefix` argument**, so it reports an empty name to the quantization machinery. The skip-list matches layers *by name*; an empty name matches nothing, so the layer gets quantized construction no matter what the list says. No config-side workaround can fix a layer that has no name.

Checking vLLM's current upstream source showed **the fix already exists in main**: a 3-line change threading the prefix through (signature, construction call, instantiation site). The image, built June 3, predates it.

**Fix:** backported upstream's 3 lines verbatim and mounted the patched file read-only over the image's copy at launch (`patches/gemma4_unified.py`; pristine original kept as `.orig` for the diff). The image itself stays untouched and pinned. The patch only affects how the layer is *named* at construction — and the layer only executes for image inputs, which are disabled — so it cannot affect text-only measurements. **Retire the patch at Week 13 version convergence**; the target version must postdate this fix landing in upstream main.

### Launch 3: clean load

```
Model loading took 8.28 GiB
Available KV cache memory: 11.82 GiB (eager) / 11.0 GiB (with CUDA graphs)
GPU KV cache size @ MML 32768: 245,222 tokens (eager) / 228,200 (CUDA graphs)
```

Notable: 8.28 GiB on-GPU is *less* than the 9.56 GiB checkpoint on disk — consistent with weight tying (the output layer sharing storage with the input embedding table at runtime), which removes one of the two ~2 GiB vocabulary tables we had budgeted separately.

## Baseline: eager vs CUDA graphs (MML 32768, util 0.90)

"Eager" means every GPU operation is launched individually by the CPU; CUDA graphs record the whole per-token kernel sequence once and replay it with a single launch, removing per-kernel CPU overhead at the cost of startup time and a memory reservation.

| Config | KV pool | KV tokens | Decode (warmed, c=1) |
|---|---|---|---|
| eager | 11.82 GiB | 245,222 | ~69 tok/s |
| CUDA graphs (production) | 11.00 GiB | 228,200 | ~78 tok/s |

- **First-probe cold-start artifact confirmed again:** the first request after every boot pays Triton kernel JIT compilation (~4.1 s vs ~2.1 s steady-state). All recorded rates use the second-or-later probe.
- **CUDA-graph memory accounting is explicit in this build:** estimated 0.79 GiB reserved ahead of the KV pool, actual capture 0.68 GiB — a 17% overshoot, far better behaved than the ~2× over-allocation characterized on v0.21.0 in Week 11. The recovery lever (`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`) would buy back ~0.79 GiB (~16K tokens at short-MML rates); not pulled — default-on is the held-constant condition.
- **Prediction miss worth recording:** CUDA graphs were predicted to lift decode to 90–110 tok/s; they delivered 78. The removed launch overhead was only ~1.7 ms/step. At ~70% of the naive memory-bandwidth bound, the remaining gap lives in the kernels themselves (forced TRITON_ATTN due to heterogeneous head dims; Marlin dequantization; bf16 lm_head) — none of which graphs touch. Lesson: eager-vs-graphs deltas are small when kernel quality, not launch overhead, binds.

## The context-ceiling walk

Walked `--max-model-len` (MML — the maximum sequence length the server will accept) across four rungs. The KV pool held essentially constant; the *token capacity* of that pool grew dramatically:

| MML | KV pool | KV tokens | Concurrency at MML | avg KB/token |
|---|---|---|---|---|
| 32,768 | 11.00 GiB | 228,200 | 6.96× | 51.8 |
| 65,536 | 11.00 GiB | 346,672 | 5.29× | 34.1 |
| 131,072 | 10.99 GiB | 467,959 | 3.57× | 25.2 |
| 262,144 | 10.99 GiB | 567,180 | 2.16× | 20.8 |

Two findings:

**1. The profiler's reservations are MML-insensitive on this build.** The pool didn't move across an 8× MML range. (Contrast with the pool-grows-with-MML behavior characterized on v0.21.0 in Week 11 — engine-version-dependent, another reason rung-walking beats extrapolation.)

**2. Per-token KV cost falls with sequence length, and a two-coefficient model fits it almost exactly.** Gemma 4 alternates two attention layer types: sliding-window (SWA) layers keep KV only for a fixed recent window regardless of sequence length, while global layers keep KV for the entire sequence. So a sequence's cost is (fixed SWA window) + (per-token global cost), and the *average* per-token cost drops as sequences get longer:

```
avg_cost(L) = 16.4 KB/token  +  1.08 GiB/sequence ÷ L

Fitted on the 65K and 131K rungs.
Held-out check at 32K: predicted 51.78 KB/token, measured 51.76 (0.04% error).
Forward check at 262K: predicted 560–580K tokens / 2.1–2.2×, measured 567,180 / 2.16×.
```

This is the quantitative form of the Week 11 "two-coefficient, regime-dependent KV cost" note — now fitted from measurement rather than estimated from the model card.

**Verdict: the single-card 12B is KV-unconstrained across its entire defined context range.** The full 262K architectural context fits with >2× concurrency in reserve. The fp8-KV-cache lever and the PP=2 two-card fallback were both prepared and **neither was needed**.

## Functional verification

At every recorded config: process placement confirmed on GPU 1 (uuid `a7370cb3`) via `nvidia-smi --query-compute-apps` uuid-join; two-probe pair (first discarded as JIT warmup) with unique nonces; coherent output (correct Rayleigh-scattering answer) and stable ~78–79 tok/s at MML 32,768 and 262,144 alike.

**Honest scope limit:** the 262K probes confirm the *configuration* is functional at short context. They do not exercise a long sequence — actually filling 100K+ tokens (prefill behavior, decode at depth, output quality) is separate future work.

## Prediction scorecard

| Prediction | Outcome |
|---|---|
| eager fixes the load OOM | **Wrong** — OOM is in weight construction, before compile/capture exist |
| Day 1 attribution (weights + 2 vocab tables + cudagraph set) | **Mostly wrong** — it was unquantized bf16 construction |
| Packed weights 9–11 GiB | Over by ~1–2 GiB (8.28; weight tying not budgeted) |
| KV pool ~10–12 GiB | Hit (11.82 / 11.0) |
| CUDA-graph cost 0.5–1.3 GiB | Hit (0.79 est / 0.68 actual) |
| Graphs lift decode to 90–110 tok/s | **Miss** — 78; kernel quality binds, not launch overhead |
| Rung-1 tokens 218–228K (fixed-cost assumption) | **Miss** — 346,672; regime-dependent cost, model corrected |
| Rung-3 (fitted model) 560–580K / 2.1–2.2× | Hit (567,180 / 2.16×) |

## Open questions / parked items

- **What does the recipe's 128K pin encode?** Not memory (262K fits easily here). Most likely a quality-validation boundary — only answerable with long-context quality evaluation, not fit-testing.
- **Long-context functional behavior** (large prefill, decode at depth) — untested.
- **Parked levers** (each a deliberate single-variable step if ever needed): zero the `video` modality to reclaim the 2,496-token encoder-cache budget; KV-tax recovery env (~0.79 GiB); fp8 KV cache (Week 11 notes say SM 8.9+ required — verify before assuming it's available on the 3090s).
- **Production MML choice** for the sub-agent worker — the walk says anything up to 262K fits; the *right* number is a workload/quality decision, not a memory one.

## Artifacts

- `patches/gemma4_unified.py` (+ `.orig`) — 3-line upstream backport, with provenance in the commit message
- `tools/start-12b-qat.sh` — temporary Week 12 launcher encoding the verified config (patch mount + full-override blob); retire with Week 13 version convergence
- `results/day2-context-ceiling-walk_gemma-4-12B-it-qat-w4a16-ct.json` — structured record of all launches, rungs, probes, and the fit
