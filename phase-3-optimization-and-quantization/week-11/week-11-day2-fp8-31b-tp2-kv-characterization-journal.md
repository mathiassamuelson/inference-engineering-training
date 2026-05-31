# Week 11 Day 2 — FP8 31B Dense, TP=2: KV characterization & single-request sweep

**Model:** `RedHatAI/gemma-4-31B-it-FP8-block`
**Config:** vLLM 0.21.0 (pinned by digest), TP=2 on the NVLink pair (GPUs 0,2), graphs ON, `--gpu-memory-utilization 0.95`, `--kv-cache-dtype auto` (resolves to BF16 on SM 8.6)
**Date:** 2026-05-31 · host `inference`

## Goal

Launch Gemma 4 31B Dense FP8 under TP=2, **characterize the achievable KV pool / context window empirically** (the 31B is weight-dominated and KV-starved, and its KV cost model does not transfer from the 26B MoE), pick a working MML, then run the standard single-request throughput sweep. Day 2 is TP=2 / c=1 only — PP and concurrent sweeps are Day 3.

## Pre-session: image digest verification

Initial check compared the wrong field. `docker images ... {{.ID}}` prints the **image config digest** (`sha256:2497255b…`), which is a different hash from the **manifest/repo digest** the Day 1 pin refers to (`a230095847…`); the two are never equal, so the comparison was meaningless. The correct check is the repo digest:

```bash
docker inspect vllm/vllm-openai:v0.21.0 --format '{{range .RepoDigests}}{{println .}}{{end}}'
```

This returned the expected `…@sha256:a230095847e93bd4df9888b33dab956fa9504537b828a23657d2b26fed57b5c9`. Pinned build confirmed present; held-constant requirement for the TP-vs-PP comparison intact.

## KV characterization

The 31B Dense has heterogeneous head dims (`head_dim=256` on 50 SWA layers, `global_head_dim=512` on 10 global layers), so per-token KV cost is two-coefficient and regime-dependent: the SWA layers cap their KV at the sliding window, while the 10 global layers grow linearly with sequence length. Two startup probes plus the Day 1 failed-boot anchor were used to fit the model. `--kv-cache-dtype auto` resolves to BF16 (FP8 KV needs SM 8.9+).

| Probe | MML | Available KV/GPU | KV pool (tokens) | Max concurrency | Implied per-seq KV |
|-------|-----|------------------|------------------|-----------------|--------------------|
| A | 8192 | 3.25 GiB | 11,722 | 1.43x | 2.27 GiB |
| B | 16384 | 3.25 GiB | 20,609 | 1.26x | 2.58 GiB |
| Day 1 (failed boot) | 131072 | 2.07 GiB* | — | — | 6.96 GiB |

\* Day 1 ran at util 0.90; Probes A/B at 0.95. The per-seq KV figures are config-independent and used directly for the fit.

**Available KV is a fixed 3.25 GiB regardless of MML** — it is bounded by weights (16.93 GiB/GPU) + CUDA graphs (0.88–0.96 GiB) + the util ceiling, not by MML. MML only sets the window into a fixed pool.

### Fitted cost model

```
per-seq KV (per GPU) ≈ 1.97 GiB  +  39.2 KiB/token × seq_len
```

- **1.97 GiB floor** = capped SWA layers + base allocation.
- **39.2 KiB/token slope** = the 10 global layers growing linearly.

Validation against all three anchors:

| seq_len | predicted | measured | error |
|---------|-----------|----------|-------|
| 8192 | 2.28 GiB | 2.27 GiB | +0.4% |
| 16384 | 2.58 GiB | 2.58 GiB | <0.1% |
| 131072 | 6.87 GiB | 6.96 GiB | −1.3% |

The two-coefficient sliding-window model holds across a 16× span in length. This is the signature the Week 11 plan predicted: a naive linear projection from the 131072 anchor overshoots short-sequence cost ~8×, because most of the per-token cost at long lengths comes from the globals, which are a minority of layers.

### Chosen working MML

Single-sequence ceiling (graphs ON, util 0.95): solving `3.25 = 1.97 + 39.2 KiB × L` gives **L ≈ 34,300 tokens** of total context (prompt + generation sharing one allocation). Chose **MML 33024** = 32768 top prompt size + 256 generation tail, sitting just under the ceiling.

Launch at MML 33024 confirmed the prediction:

```
GPU KV cache size           33,349 tokens   (predicted ~34,300, within 3%)
Max concurrency @33,024     1.01x           (one full-window sequence fits — required for c=1)
```

**65536 was dropped:** a 65536+256 sequence needs ~4.43 GiB > 3.25 GiB available — it will not hold even a single sequence in this config. This is a KV-capacity finding, not a tuning miss. The fix (PP, eager mode, or `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`) was deliberately not applied here to keep the config held-constant for the comparison.

## Single-request throughput sweep (TP=2, c=1)

`tools/throughput_sweep.py` schema v3, `--concurrency 1 --iterations 3 --warmup 1`, prompt sizes `512 2048 4096 8192 16384 32768`, model name passed explicitly. 18/18 requests OK, zero failures.

| Prompt size | Prefill tok/s | Decode tok/s | TTFT (s) | Gen throughput tok/s | Wall (s) |
|------------:|--------------:|-------------:|---------:|---------------------:|---------:|
|   512 | 1933.0 | 44.22 |  0.264 | 42.45 |  6.03 |
|  2048 | 1850.4 | 43.36 |  1.104 | 36.65 |  6.99 |
|  4096 | 1781.1 | 42.16 |  2.294 | 30.69 |  8.34 |
|  8192 | 1678.6 | 40.11 |  4.868 | 22.81 | 11.23 |
| 16384 | 1525.0 | 38.15 | 10.718 | 14.71 | 17.40 |
| 32768 | 1291.5 | 34.86 | 25.312 |  7.85 | 32.63 |

Per-size stdevs were tight throughout (decode σ < 0.02 tok/s, prefill σ < 2 tok/s), so the curve is clean, not noisy.

**Decode tok/s vs Gen throughput tok/s** — these measure different things and their divergence is informative:

- **Decode tok/s** is the steady-state generation rate measured over the **decode phase only** (`gen_tokens / decode_time`, excluding prefill/TTFT). It answers "once the model starts emitting tokens, how fast?" and degrades only mildly with context (44.22 → 34.86, −21%) as global-layer KV attention grows.
- **Gen throughput tok/s** is the **end-to-end** effective rate (`gen_tokens / total_wall_time`, prefill included). It answers "tokens per second the caller actually experiences for the whole request" and collapses hard with context (42.45 → 7.85, −82%) because prefill (TTFT) increasingly dominates the wall clock.

At 512 the two nearly coincide (44.22 vs 42.45) — prefill is a rounding error on a 6 s request. At 32768 they diverge ~4.4× (34.86 vs 7.85) because 25.3 s of the 32.6 s wall is prefill. The gap between the columns *is* the prefill tax, and it is the headline cost story for long-context single-request serving on this model.

## Cross-checks against Day 1 anchors

**1. Decode @2048 — passes.** 43.36 tok/s vs Day 1 smoke 43.4. Clean tie-in.

**2. Prefill @2048 — passes.** 1850 tok/s vs Day 1 1858.

**3. Prefill "rise then plateau" — reconciled, not a discrepancy.** The cross-check note predicted a rise-then-plateau shape; the actual curve from 512 up is a **monotonic decline** (1933 → 1291, −33%). These are consistent once the ranges are aligned: the rise the note described was the 128→512 fixed-overhead amortization visible in Day 1's truncated range (1749 → 1840 → 1858). By 512 we are already past that knee; from there, O(n²) prefill attention on the 10 global layers dominates and the rate falls. The "plateau" was the ceiling of Day 1's prompt-size range, not a real plateau. TTFT confirms the quadratic signature — growth per doubling is super-linear and accelerating (×2.12, ×2.20, ×2.36 across the top three sizes).

**Decode degradation:** decode eases ~21% (44.22 → 34.86) as context grows, consistent with the same global-layer KV attention cost climbing with sequence length.

## Validity notes

- `cached_tokens` returned **`null`** on every request (not the `1` the pickup doc anticipated for BOS-only). This appears to be vLLM 0.21.0 not populating the field rather than a cache hit. The nonce prefix is confirmed working independently: per-request `prompt_tokens` jitters by size (511/510/510, 2044/2042/2044, …), which only happens if each request gets a distinct nonce prefix. No request approached the contamination threshold. Data is trustworthy; flagging the `null` so it isn't misread as a cache miss later.

## Corrections / clarifications logged this session

- **"Top runnable prompt size" → context-window cap.** The 34,300-token ceiling is total sequence length (prompt + generation), not a prompt-size limit. A 32768-token prompt leaves ~1,500 tokens of generation headroom; 32768+256 fits. Correct framing: achievable context window ≈ 34K, top sweep prompt 32768 chosen to sit under it with a generation tail.
- **PP=2 does not halve weights again.** The 16.93 GiB/GPU is already the post-TP=2 figure (~31 GiB FP8 weights sharded across the pair). PP=2 partitions by layers across the same 2 GPUs → same ~50% per GPU, not 25%. Only PP=4 quarters it. The earlier "PP=2's pool dwarfs TP's" claim was built on the wrong number and is retracted. At equal per-GPU weights, PP=2's KV pool should be in the same ballpark as TP=2's; any material divergence would be a second-order (per-stage activation / pipeline buffer / graph memory) effect — and is a Day 3 finding to surface, not prejudge.

## Deferred levers (not applied — would break held-constant config)

- **Video encoder budget.** Despite `--limit-mm-per-prompt '{"image":0,"audio":0}'`, vLLM still profiled 1 video item and reserved a 4096-token encoder cache. Adding `"video":0` is a real KV-reclaim lever but must be tested separately and, if adopted, held constant across TP=2 and PP=2.
- **CUDA-graph memory tradeoff.** Two distinct levers, not interchangeable: `--enforce-eager` reclaims graph memory but kills graph decode (real per-step penalty, would depress every number); `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` reclaims only the profiler's conservative reservation while keeping graph decode (cheaper, risks capture-time OOM). The latter currently can't be tested because `start-vllm.sh` only forwards `HF_TOKEN` — a generic `--env KEY=VAL` passthrough is a pending script change.

## Deliverables

- Working MML **33024**, KV pool **33,349 tokens**, max concurrency **1.01x** at the chosen config — the per-config "KV pool capacity" entry for the eventual TP-vs-PP table.
- Single-request TP=2 sweep JSON committed to `phase-3-optimization-and-quantization/week-11/results/`.
- This journal.

## Status: Day 2 complete

KV characterization done, working MML chosen and validated empirically, c=1 sweep run and cross-checked.

## Open threads for Day 3 (no work started)

- **PP=2 on the NVLink pair**, same held-constant config (graphs ON, util 0.95). Run the same KV characterization first — do not assume the TP pool transfers; compare per-GPU KV pool capacity head-to-head and surface any asymmetry.
- **Concurrent sweeps (c>1).** The KV pool is small (~33K tokens, ~1 full-window seq), so concurrency at large prompt sizes will be KV-capacity-bound, not compute-bound, on this model — the concurrent ceiling is itself a measurement.
- Whichever graph/memory config is chosen for the comparison must stay identical across TP=2 (Day 2) and PP=2 (Day 3).
