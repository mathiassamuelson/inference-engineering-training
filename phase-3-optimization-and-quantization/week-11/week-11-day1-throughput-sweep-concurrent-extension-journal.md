# Week 11 Day 1 — Concurrent-load extension of `throughput_sweep.py` (schema v3)

**Date:** 2026-05-30 / 2026-05-31 (MT)
**Phase:** 3 (Optimization & Quantization, Weeks 11–14) / Week 11 (TP-vs-PP comparison thread)
**Host:** `inference` (4× RTX 3090; GPUs 0+2 NVLink, 1+3 on PCIe 3.0 x1)
**vLLM:** `vllm/vllm-openai:v0.21.0` (digest `sha256:a230095847e93bd4df9888b33dab956fa9504537b828a23657d2b26fed57b5c9`)

## Goal

Day 1 is engineering, not inference runs: extend `tools/throughput_sweep.py` to drive
concurrent load, since concurrent throughput is the regime where pipeline parallelism's
story becomes legible and is therefore the gating dependency for the rest of Week 11.
Single-request-first-then-circle-back would double the run work later.

## Pre-session checks

- **vLLM PR #42630 relevance — resolved: stay on 0.21.0.** The "Gemma 3/4 multi-GPU fix"
  targets Gemma **3n** multimodal (`gemma-3n-E4B-it`): it adds `input_is_parallel=False`
  to a `RowParallelLinear` in the multimodal→text embedding projection (`gemma3n_mm.py`),
  fixing a `mat1/mat2` dim mismatch under TP>1. The repro models were 3n multimodal
  variants; the touched path is the MM module. Week 11 runs a dense text model with
  `--limit-mm-per-prompt '{"image":0,"audio":0}'`, so the path is not exercised. No 0.22
  upgrade; no second digest to track.
- **Storage:** `~/.cache/huggingface` on a 1.8T volume, 1.2T free. Ample for the ~31 GB FP8 model.
- **Image:** `v0.21.0` present, digest matches the pinned value exactly.
- **Port 8000:** `irs-vllm` not running; clear.
- **Image housekeeping:** removed `vllm/vllm-openai:v0.20.0` (unreferenced). **Kept**
  `vllm/vllm-openai:gemma4` (`sha256:0cb12dc9…`) — it is the pinned pre-HMA baseline for
  the still-unwritten before/after KV-fix Pulse article (vllm#39133). Deleting it would
  discard the only reproducible "before" case.

## What changed in the script

Extended `tools/throughput_sweep.py` from schema v2 (single-request) to **schema v3**
(single-request or concurrent). Key design decisions:

1. **Single async implementation for all concurrency levels.** The request path was
   converted from synchronous `requests` to `asyncio` + `httpx.AsyncClient`. At
   `--concurrency 1` this is a gather-of-one, reducing exactly to v2's
   one-request-per-iteration. The motivation is measurement integrity: the most important
   within-config comparison this week is single-request vs concurrent, and keeping c=1 and
   c=N on *different* measurement code would introduce a measurement-path confound into
   that comparison. One code path removes it. The cost — c=1 numbers now come from async
   code rather than the proven sync path — is exactly what the backward-compat smoke test
   was designed to retire (see Validation).

2. **Wave model.** A "wave" is N requests dispatched simultaneously and gathered.
   `--concurrency N` sets the wave width; `--iterations` is now the number of measured
   waves per prompt size; `--warmup` is discarded waves.

3. **Aggregate computed from per-request records, never measured separately.** Per-wave
   aggregate generation throughput =
   `sum(completion_tokens over OK requests) / (max completion_time − min dispatch_time)`.
   A shared per-wave monotonic epoch makes dispatch/completion times mutually comparable.
   This denominator includes prefill time, so it is a *system throughput* figure, distinct
   from per-request `decode_rate` (which excludes prefill).

4. **Per-request records preserved** with `request_id`, `dispatch_time`, `completion_time`
   added to all existing v2 fields. Each c=1 record is a strict superset of a v2 iteration
   record.

5. **Default filename gains `_c<N>_`** so multiple concurrency runs against the same model
   never silently overwrite (consistent with the existing model-name-in-filename rule).

6. **Schema v3 documented in the script docstring** (concurrency model, aggregate formula,
   and the client-side TTFT-under-load caveat below). No separate README needed.

7. **HTTP client migration** `requests` → `httpx`. `requests` is no longer imported by this
   script (but remains a repo dependency — see Housekeeping).

### Documented caveat: per-request TTFT/prefill under concurrency is an *observed* quantity

With N coroutines on one event loop, the instant the client *observes* a request's first
token can be delayed by the loop servicing siblings. At N=1 this is negligible. At N>1,
per-request TTFT (and the `prefill_rate` derived from it) reflects genuine server
contention **plus** a client-side observation artifact, and should be read as "observed
TTFT under load," not isolated prefill latency. The **aggregate** throughput figure is
immune — it is derived from token counts and wave wall-clock, not from individual TTFTs.
This is documented in the docstring and was directly observed in the c=4 validation
(below), confirming it behaves as predicted rather than as a measurement bug.

## Smoke-test server

The default launch (FP8 31B, TP=2, MML 131072) **failed at KV allocation**, not OOM:
weights loaded fine (~16.9 GiB/GPU), but after weights + CUDA graphs only ~2.07 GiB/GPU
remained for KV, while one sequence at MML 131072 needs ~6.96 GiB. vLLM's own estimated
max length at these settings was ~4912 tokens.

This is structural, not a misconfiguration: the 31B Dense under TP=2 is **weight-dominated
and KV-starved** — ~17 of 24 GiB/GPU is weights. It is the opposite of the AWQ-INT4 MoE
(small per-GPU weights → ~891K-token pool at MML 262144). **The 131072 placeholder and the
MoE's 262144 target do not transfer to this model.** Achievable MML and the concurrent-
sequence ceiling are Day 2 characterization questions.

For the smoke test (which does not depend on MML), relaunched at `--max-model-len 4096
--gpu-mem-util 0.95`, which boots comfortably. Note: the 31 GB FP8 download completed
during the failed boot, so Day 2's first task is already done.

## Validation

A new `start-vllm.sh` (parameterized over model/mode/size; default = FP8 31B, TP=2, GPUs
0,2) was also written this session to launch the experiment configs; it was used for the
smoke server. Validation of the script extension proceeded on three axes.

### 1. Decode, c=1 — matches v2 across all sizes
The steady-state long-window measurement, least sensitive to the HTTP-path change.

```
size   v2 decode tok/s   v3 decode tok/s     delta
 128         44.78            44.90          +0.3%
 512         44.18            44.28          +0.2%
2048         43.48            43.39          -0.2%
```

### 2. Prefill, c=1 — matches v2 in the compute-bound regime
At 128 tokens prefill is trivial and TTFT is mostly fixed per-request overhead, so
"prefill rate" there is overhead-dominated and diverged ~26% (v2 1393 vs v3 1749) — an
artifact below the experiment's prompt-size range, not a data discrepancy. By 2048 (firmly
compute-bound, the regime the real 4K–64K comparisons live in) the two agree tightly:

```
size   v2 prefill tok/s   v3 prefill tok/s    delta     v3 stdev
 512        1828              1840            +0.7%      (noisy)
2048        1867.9            1857.9          -0.54%     1.1 tok/s (0.06%)
```

The 2048 agreement sits on internally tight data on both sides (v2 stdev 4.4, v3 stdev
1.1 tok/s), so it is two clean measurements landing on the same value, not two noisy ones
overlapping. Tokenizer calibration (`chars_per_filler_token` = 5.986) was identical across
v2 and v3, confirming the tokenizer path is unchanged by the rewrite.

### 3. Concurrency path, c=4 — the new capability
4/4 OK on every wave at both 128 and 512, zero failures. Aggregate throughput stable and
~3.7× the single-request aggregate, confirming continuous batching:

```
size   c=1 aggregate tok/s   c=4 aggregate tok/s   speedup   c=4 agg stdev
 128        ~44.5                 163.6              3.7x       0.48
 512        ~42.4                 141.1              3.3x       0.26
```

Per-request `prefill_rate` within a wave spread widely (512 wave 0: 1787 / 937 / 485 / 481
tok/s; summary stdev ~591). This is **not** measurement breakage — it is the documented
client-side observation artifact. The tell: all four requests dispatched within ~0.5 ms,
but observed TTFTs stair-step (0.29 → 0.55 → 1.06 → 1.06 s) in dispatch order, i.e. the
single event loop drains one stream's first token before the next. The aggregate (built
from token counts + wave wall-clock, not TTFTs) is unaffected and is the figure the
experiment will report. Decode spread (512: 36.7–41.2 tok/s) is partly the same effect and
partly real GPU-sharing contention; again, the per-request lens, not the aggregate.

## Commit

v3 swapped into `tools/throughput_sweep.py` and committed. Post-swap the script imports
`httpx` and no longer `requests`.

## Housekeeping: `requests` stays installed

After the swap, repo scan (venv excluded) for `import requests`:

```
./phase-2-production/week-08/exp2_throughput_sweep.py
./phase-2-production/week-06/triton_embedding_test.py
```

`requests` remains a live dependency of two other repo scripts, so it was **not**
uninstalled from the `ai-inference` venv. (`httpx` 0.28.1 was added to the venv for the new
script.) Note for a future housekeeping session, not Day 1 scope: `week-08/
exp2_throughput_sweep.py` is the predecessor that `tools/throughput_sweep.py` superseded;
if retired, `requests` would drop to a single user.

## Migration note for downstream analysis (action item, not Day 1)

Schema v3 reorganizes each prompt-size entry around `waves[]` rather than v2's flat
`iterations[]`. All v2 *information* is preserved (per-request records are supersets;
v2's per-size `summary` now lives under `summary.per_request`), but **any analysis code
that reads `results[i]["iterations"]` must switch to
`results[i]["waves"][w]["requests"]`.** This is relevant because `week-08/
exp2_throughput_sweep.py` and any Week 8/9 notebooks parsing v2 output predate the new
layout. Flagging so a v2-shaped parser isn't silently pointed at a v3 file later.

## Day 1 status: complete

- [x] PR #42630 triaged → stay on 0.21.0
- [x] Homework (storage / image digest / port) cleared
- [x] `--concurrency N` added; async/httpx single path; schema v3
- [x] Schema v3 documented in docstring
- [x] Smoke-tested c=1 (backward compat vs v2) and c=4 (concurrency path)
- [x] Committed to `tools/throughput_sweep.py`
- [x] `requests` cleanup resolved (keep)

## Open threads for Day 2 (no work started)

- **Characterize achievable MML for FP8 31B under TP=2.** The 131072 placeholder is far
  too high; capture the `GPU KV cache size` startup line across a couple of MML settings
  and fit empirically. Do **not** reuse the 26B MoE coefficients — the 31B has
  heterogeneous head dims (`head_dim=256`, `global_head_dim=512`) and 50 SWA + 10 global
  layers, so per-token KV cost is two-coefficient and regime-dependent (SWA layers cap at
  the window; globals grow linearly). The failed-boot anchors (6.96 GiB/seq at MML 131072;
  ~4912 est. max len at 2.07 GiB free) already show the sliding-window signature — the
  linear projection (~39K tokens) overshoots the estimate ~8×.
- **KV-starvation bears on Day 3 concurrent sweeps:** the small pool will limit concurrent
  sequences, so PP's continuous-batching regime may be KV-capacity-bound rather than
  compute-bound on this model. Levers to explore Day 2: `--gpu-mem-util` toward 0.94–0.95;
  the CUDA-graph memory tradeoff (profiler costs ~0.04 effective util;
  `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` or `--enforce-eager` reclaims it for KV at a
  speed cost).
- Day 2 proper: launch FP8 31B TP=2, KV characterization, single-request sweep.
