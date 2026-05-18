# Week 9 Day 4: vLLM 0.21.0 HMA before/after

**Date:** 2026-05-17
**Hardware:** 4× RTX 3090 (24GB), TP across GPUs 0 and 2 via NVLink
**Model:** `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` (Gemma 4 26B A4B MoE, AWQ-INT4)

## Context

Week 9 was paused at Day 3 with an open upstream dependency: [vllm-project/vllm#39133](https://github.com/vllm-project/vllm/issues/39133). vLLM's KV cache manager was not exploiting Gemma 4's hybrid sliding-window attention architecture; all 30 layers were being sized at `max_model_len` regardless of layer type. Gemma 4 26B A4B follows the standard Gemma 3/4 5:1 ratio pattern: **25 sliding-window layers (window=1024) and 5 global layers**, with the final layer always global. The old build's failure to distinguish these produced ~5.5× more KV memory per token of capacity than llama.cpp on the same model, and made the Day 1/Day 2 framework comparison a "framework + bug" comparison rather than a framework comparison.

vLLM 0.21.0 (released 2026-05-15) lands the Hybrid Memory Allocator (HMA) via PRs [#41228](https://github.com/vllm-project/vllm/pull/41228), [#41445](https://github.com/vllm-project/vllm/pull/41445), and [#39571](https://github.com/vllm-project/vllm/pull/39571), which addresses the core symptom in #39133. Today's session re-tests against 0.21.0 to characterize the fix and re-run the single-request throughput sweep.

## Stack

| Component | Old build (Day 1) | New build (today) |
|---|---|---|
| vLLM image | `vllm/vllm-openai:gemma4` | `vllm/vllm-openai:v0.21.0` |
| vLLM version | 0.18.2rc1.dev73+gdb7a17ecc | 0.21.0 |
| Image digest | `sha256:0cb12dc9...4922834` | `sha256:a2300958...fed57b5c9` |
| HMA | not enabled | enabled |
| KV cache dtype | bf16 (auto) | bf16 (auto) |

llama.cpp side (Day 2 data) unchanged and not re-collected — the upstream fix is vLLM-only.

## Per-layer K/V cost reference

From the Gemma 4 26B A4B `text_config`:
- SWA layers (25 of 30): `num_key_value_heads=8`, `head_dim=256`, K and V stored separately
- Global layers (5 of 30): `num_global_key_value_heads=2`, `global_head_dim=512`, `attention_k_eq_v=true` (K and V are the same tensor)

Per-token, per-layer, per-GPU cost (BF16, TP=2) with the formula `(num_kv_heads/TP) × head_dim × K_V_factor × sizeof(dtype)`:

| Layer type | Calculation | Bytes/token/layer/GPU |
|---|---|---:|
| SWA | (8/2) × 256 × 2 × 2 | 4,096 |
| Global (true, with K=V) | (2/2) × 512 × 1 × 2 | 1,024 |
| Global (if K, V stored separately) | (2/2) × 512 × 2 × 2 | 2,048 |
| Global (if SWA-shaped) | (8/2) × 256 × 2 × 2 | 4,096 |

The "true global" cost is what an allocator that exploits the architecture fully would achieve. The three "if" rows define the structural scenarios for what HMA might actually be doing — discriminated by the sweep data below.

## Step 1: KV cache allocation sweep

### Method

Launched vLLM 0.21.0 four times at `--max-model-len` of {262144, 131072, 32768, 8192} with `--gpu-memory-utilization 0.90`, capturing the `GPU KV cache size` line and `Available KV cache memory` line from each startup log. Raw logs at `phase-2-production/week-09/results/week09-vllm-021-kv-sweep/`. Available KV memory was 10.17 GiB per GPU at every setting.

### Results

| `--max-model-len` | KV cache tokens | Amortized B/token | Max concurrency |
|---:|---:|---:|---:|
| 8,192 | 146,668 | 74,448 | 17.90× |
| 32,768 | 415,501 | 26,281 | 12.68× |
| 131,072 | 766,222 | 14,252 | 5.85× |
| 262,144 | 891,535 | 12,249 | 3.40× |

For reference, the old `:gemma4` build (Day 3 sweep) was flat at ~95,472 tokens across this same range, with amortized per-token cost ~122,925 B/token regardless of `max_model_len`. At MML=262144, the 0.21.0 number represents a **9.3× increase** in KV pool capacity for the same VRAM budget.

### Analysis

The pre-test prediction (in the session pickup doc) was that HMA would *reduce* per-token cost as MML increased, because freed SWA over-allocation would shrink. That intuition was wrong. The observed curve is monotonic the other way: capacity grows with MML, and amortized per-token cost falls.

The reason becomes clear when the data is modeled. With 25 SWA layers capped at the window (1024 tokens) regardless of MML, and 5 global layers that scale with MML, the per-sequence KV cost should be:

```
per_sequence(MML) = base_overhead + 25 × 1024 × 4096 + 5 × MML × g
                  = base_overhead + 100 MiB + 5·g·MML
```

where `g` is the unknown per-token-per-layer-per-GPU cost for globals. Total KV tokens at a given MML is then `available_mem × MML / per_sequence(MML)`.

Solving the two-equation system from MML=262144 and MML=8192:

```
base + 100 MiB + g·5·262144 = 12,249 × 262,144 = 3,210,665,856 bytes
base + 100 MiB + g·5·  8192 =  74,448 ×   8,192 =   609,898,496 bytes
                            difference =          2,600,767,360 bytes
                                       / (5 × 253,952)
                                  ⇒ g = 2,048 B/token/layer/GPU
                               base ≈ 400 MiB
```

Verifying this fit against the two intermediate data points:

| `--max-model-len` | Predicted amortized B/token | Observed | Error |
|---:|---:|---:|---:|
| 8,192 | 74,453 | 74,448 | <0.1% |
| 32,768 | 26,294 | 26,281 | <0.1% |
| 131,072 | 14,254 | 14,252 | <0.1% |
| 262,144 | 12,249 | 12,249 | 0.0% |

The linear-in-MML model with `g = 2,048` and `base ≈ 400 MiB` reproduces the sweep to within 0.1% at every point. This is a strong fit — strong enough to be confident in the structural reading.

### What the fit tells us

1. **HMA correctly distinguishes SWA from global layers.** The SWA layers are being capped at the window (1024 tokens) regardless of MML, as the model assumes. Without this, the linear-in-MML form wouldn't hold; we'd see flat behavior like the old build.

2. **Global layers are sized at true global dimensions (2 heads, 512 head_dim), not SWA dimensions.** The fitted `g = 2,048` matches the "K and V stored separately at global dimensions" row in the reference table above. The "globals stuck at SWA-shape" scenario would have produced `g = 4,096` and asymptotic amortized cost of ~20,480 B/token — that's not what we see.

3. **The K=V unification optimization is NOT being applied to global layers.** Gemma 4's `attention_k_eq_v=true` flag indicates K and V are the same tensor in global layers. If HMA exploited this, `g` would be 1,024 and asymptotic amortized cost would be ~5,120 B/token. The fitted `g = 2,048` says K and V are being stored as separate tensors despite the architectural opportunity to share them. This leaves a clean 2× over-allocation on global layers — roughly 5,120 B/token of avoidable cost at the MML=262144 asymptote.

4. **There is a ~400 MiB fixed per-sequence overhead.** Independent of MML and presumably independent of sequence length. Candidates: CUDA graph buffers, block-level page rounding (HMA's uniform page size constraint applied to the small global-layer pages), per-sequence activation reserves. The journal can't identify which without more investigation, but the term is clearly present in the data. It dominates amortized cost at low MML (74K B/token at MML=8192 is mostly base overhead, not real KV) and becomes negligible at high MML.

The net story: HMA delivers most but not all of the available KV efficiency. The unfinished win is **K=V unification on globals**, which would roughly double KV pool capacity again at MML=262144 (from ~891K to ~1.7M tokens).

### Practical read

For statmon-ai's operational window (7–15K tokens per session):

| `--max-model-len` | Sessions concurrently held in KV pool (15K each) |
|---:|---:|
| 8,192 | cap below session size — disqualified |
| 32,768 | 27 |
| 131,072 | 51 |
| 262,144 | 59 |

The concurrency ceiling at MML=262144 (~59 simultaneous 15K-token sessions) is well past any realistic load profile.

## Step 2: Single-request throughput sweep on vLLM 0.21.0

### Method

Used `tools/throughput_sweep.py` (schema v2) with the same parameters as Day 1: `--prompt-sizes 512 2048 4096 8192 16384 32768 65536 --max-tokens 256 --iterations 3 --warmup 1`. Nonce-prefixed prompts to defeat prefix caching. Server: vLLM 0.21.0, TP=2 on GPUs 0+2, `--max-model-len 262144 --gpu-memory-utilization 0.90 --max-num-batched-tokens 4096`.

Results JSON: `phase-2-production/week-09/results/throughput_sweep_vllm-openai_gemma-4-26B-A4B-it-AWQ-4bit_20260517T175943Z.json`.

### Results

| Prompt tokens | Prefill (tok/s) | Decode (tok/s) |
|---:|---:|---:|
| 512 | 8,498 | 160.2 |
| 2,048 | 9,851 | 154.5 |
| 4,096 | 9,846 | 147.7 |
| 8,192 | 9,257 | 136.1 |
| 16,384 | 8,216 | 125.1 |
| 32,768 | 6,626 | 108.4 |
| 65,536 | 4,766 | 94.6 |

Per-iteration values agreed within 0.5% on decode and within 5% on prefill at all sizes except two transient prefill anomalies — see note below. `cached_tokens` warnings absent throughout.

### Comparison vs old build

For full numeric side-by-side, the Day 1 JSON (`...20260410T020153Z.json`) and today's JSON are co-located. Qualitatively:

- **Decode shape essentially unchanged.** Today's decline 512→65536 is 41% (160.2 → 94.6). The Day 1 build declined ~37% across the same range. Within run-to-run noise; the HMA fix did not move single-request decode.
- **Prefill ceiling sits at ~9,850 tok/s around the 2K–4K range,** same general shape as Day 1.

### Interpretation

The #39133 bug was in KV cache *sizing*, not in attention math. The old build over-reserved memory by treating all layers as global-shaped during allocator bookkeeping, but during attention computation it still respected each layer's true attention pattern — per-step bandwidth at decode time was already correct. That's why single-request decode is preserved across the fix.

What HMA actually unlocks is the **concurrency ceiling**. The KV pool went from ~95K tokens to ~891K tokens (9.3×) at MML=262144. This has zero effect on a single sequence but multiplies how many sequences can occupy the pool simultaneously. The original Week 9 Day 4 plan — concurrent benchmarking — is now the experiment that will produce real before/after movement.

### Anomalies

Two prefill outliers in the earlier (truncated) sweep run:
- `prompt_size=128 iter 3`: 1,635 vs 4,580/4,597 in the prior iters
- `prompt_size=1024 iter 3`: 2,705 vs 9,538/9,537

Both are prefill-only — decode rates on those same iterations are identical to siblings, ruling out server-wide stall. The pattern (third iteration at small/medium prompts) suggests SSE buffering hiccups in the TTFT-based prefill estimate rather than real server slowdowns. No server-side prefill cross-check exists on vLLM to confirm; this remains an open methodology gap. The full sweep run (the JSON of record) did not show these outliers at the prompt sizes that were retained (512 and up).

## Findings

1. **HMA delivers 9.3× KV pool capacity at MML=262144** (95,472 → 891,535 tokens) on Gemma 4 26B MoE with no change in VRAM budget.
2. **HMA correctly distinguishes SWA from global layers and applies true global dimensions.** Fitted per-global-layer cost is exactly 2,048 B/token/GPU, matching the `(num_global_kv_heads/TP) × global_head_dim × K-and-V-separate × BF16` calculation.
3. **K=V unification on global layers is not applied.** Gemma 4 sets `attention_k_eq_v=true` on global layers, but HMA stores K and V as separate tensors anyway. This leaves a clean 2× over-allocation on globals — about 5,120 B/token of avoidable cost at MML=262144. If applied, KV capacity at MML=262144 would roughly double again to ~1.7M tokens.
4. **There is a ~400 MiB fixed per-sequence overhead** that dominates amortized cost at low MML and is negligible at high MML. Source unidentified — candidates include CUDA graph buffers, HMA page-size rounding, and per-sequence activation reserves.
5. **The linear-in-MML cost model `per_sequence(MML) = 400 MiB + 100 MiB + 10,240 × MML` fits the sweep to within 0.1% at every point.** Strong enough fit to be confident in the structural reading.
6. **Single-request throughput is essentially unchanged across the fix.** The bug was in pool sizing (allocator bookkeeping), not in attention math. Per-step bandwidth was already correct on the old build.
7. **The concurrent-load benchmark is where the before/after story will show.** Old build = ~95K KV tokens = severe concurrency cap. New build = ~891K = practically unbounded for this workload.
8. **The pre-test prediction about MML-vs-capacity shape was wrong** (and the journal records the wrong prediction for future-self honesty). HMA caps SWA layers regardless of MML; MML scales only the global-layer footprint, so total capacity grows monotonically with MML.

## Files produced

- `phase-2-production/week-09/results/throughput_sweep_vllm-openai_gemma-4-26B-A4B-it-AWQ-4bit_20260517T175943Z.json` — full sweep, schema v2
- `phase-2-production/week-09/results/week09-vllm-021-kv-sweep/` — KV characterization run
  - `summary.txt` — extracted KV cache lines from all four launches
  - `vllm-021-mml{8192,32768,131072,262144}.log` — full vLLM startup logs

## What this leaves open

- **K=V unification on global layers in HMA.** The 2× over-allocation is identifiable from the sweep data alone. Worth reading the HMA source to determine whether this is a deliberate non-implementation (e.g., the page-size constraint forces uniform allocation across layer types) or an unimplemented optimization that could be filed as a follow-up issue.
- **The ~400 MiB fixed per-sequence overhead.** Source not identified. Probably worth a separate investigation pass — would significantly improve max concurrency at small MML if reducible.
- **Concurrent benchmarking (the original Week 9 Day 4 item).** Now that the KV pool is right, this measures what it's supposed to measure. Requires extending `tools/throughput_sweep.py` for concurrent dispatch (configurable concurrency, per-request timings preserved, aggregate throughput computed). Deferred to a future session per the one-experiment-at-a-time discipline.
- **vLLM has no server-side prefill cross-check.** The prefill numbers are still probably lower bounds by an unknown amount. Prometheus-metrics-based measurement remains a candidate.

## References

- Upstream issue: [vllm-project/vllm#39133](https://github.com/vllm-project/vllm/issues/39133)
- HMA design doc: [docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)
- Gemma 4 architecture analysis (layer ratios, K/V dimensions): [kaitchup.substack.com/p/gemma-4-31b-and-26b-a4b-architecture](https://kaitchup.substack.com/p/gemma-4-31b-and-26b-a4b-architecture)
- Our contribution to the issue: [#issuecomment-4232552320](https://github.com/vllm-project/vllm/issues/39133#issuecomment-4232552320)
- Day 3 journal: `week-09-day3-gemma4-kv-sizing-reproduction-journal.md`
