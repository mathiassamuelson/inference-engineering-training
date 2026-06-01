# Week 11 Day 2 — FP8 31B Dense, TP=2: KV characterization & single-request sweep

**Model:** `RedHatAI/gemma-4-31B-it-FP8-block`
**Config:** vLLM 0.21.0 (pinned by digest), TP=2 on the NVLink pair (GPUs 0,2), graphs ON, `--gpu-memory-utilization 0.95`, `--kv-cache-dtype auto` (resolves to BF16 on SM 8.6), **text-only** (`--limit-mm-per-prompt '{"image":0,"audio":0,"video":0}'`)
**Date:** 2026-05-31 · host `inference`

## Goal

Launch Gemma 4 31B Dense FP8 under TP=2, **characterize the achievable KV pool / context window empirically** (the 31B is weight-dominated and KV-starved, and its KV cost model does not transfer from the 26B MoE), pick a working MML, then run the standard single-request throughput sweep. Day 2 is TP=2 / c=1 only — PP and concurrent sweeps are Day 3.

## Deployment posture: text-only

This is a text-only deployment, so the server launches with all multimodal limits zeroed. The checkpoint is a `Gemma4ForConditionalGeneration` model that carries a vision encoder; in text-only mode vLLM reports `All limits of multimodal modalities supported by the model are set to 0, running in text-only mode` and **does not load the vision tower**. Consequences, both measured below:

- **Weights load at 15.85 GiB/GPU**, not the ~16.9 GiB a multimodal load takes — the vision tower (~1.08 GiB/GPU at TP=2) is never instantiated.
- **No encoder cache budget is reserved**, so that memory stays in the KV pool.

For a multimodal checkpoint served text-only, zeroing the mm limits is therefore a real memory win on two axes (weights + KV budget), not just a request-validation flag. Held constant across the whole TP-vs-PP comparison.

### Background: what the "vision tower" is (and why dropping it is free for us)

A multimodal LLM is a text transformer with extra front-ends bolted on for non-text inputs. The image path works like this: an image is split into patches; the patches run through a dedicated transformer stack (a ViT-style encoder, SigLIP-style in the Gemma family) that has its own pretrained weights — this stack is the **vision tower**; a small **projector** MLP maps the encoder's output features into the language model's token-embedding space; those projected vectors are spliced into the token stream alongside the normal text tokens. The combined sequence then flows through the shared **LLM backbone** (the language decoder), which attends across both modalities and generates text. Audio, when supported, gets an analogous encoder feeding the same projector-into-backbone path. The name "tower" is a holdover from two-tower / dual-encoder designs where each modality had its own parallel stack before the representations met.

`Gemma4ForConditionalGeneration` is exactly this combined object: vision encoder + projector + Gemma language model. When all mm limits are 0, vLLM never instantiates the vision tower — there's no reason to load an encoder whose outputs can't enter the forward pass — which is the ~1.08 GiB/GPU weight saving plus the freed encoder cache budget.

The detail that matters for this characterization: the heterogeneous head dims driving the KV cost model (`head_dim=256` on the SWA layers, `global_head_dim=512` on the globals) are properties of the **LLM backbone's** attention, not the vision tower. So dropping the tower raised the available-memory floor (more room for KV) but left the per-token KV coefficients untouched — which is precisely what the Probe A/B reruns confirmed (same `1.97 GiB + 39.2 KiB/token`, only the 3.25 -> 4.04 GiB floor moved).

## Pre-session: image digest verification

Initial check compared the wrong field. `docker images ... {{.ID}}` prints the **image config digest** (`sha256:2497255b...`), which is a different hash from the **manifest/repo digest** the Day 1 pin refers to (`a230095847...`); the two are never equal, so the comparison was meaningless. The correct check is the repo digest:

```bash
docker inspect vllm/vllm-openai:v0.21.0 --format '{{range .RepoDigests}}{{println .}}{{end}}'
```

This returned the expected `...@sha256:a230095847e93bd4df9888b33dab956fa9504537b828a23657d2b26fed57b5c9`. Pinned build confirmed present; held-constant requirement for the TP-vs-PP comparison intact.

## KV characterization

The 31B Dense has heterogeneous head dims (`head_dim=256` on 50 SWA layers, `global_head_dim=512` on 10 global layers), so per-token KV cost is two-coefficient and regime-dependent: the SWA layers cap their KV at the sliding window, while the 10 global layers grow linearly with sequence length. Two startup probes plus the Day 1 failed-boot anchor were used to fit the model. `--kv-cache-dtype auto` resolves to BF16 (FP8 KV needs SM 8.9+).

| Probe | MML | Available KV/GPU | KV pool (tokens) | Max concurrency | Implied per-seq KV |
|-------|-----|------------------|------------------|-----------------|--------------------|
| A | 8192 | 4.04 GiB | 14,561 | 1.78x | 2.27 GiB |
| B | 16384 | 4.04 GiB | 25,601 | 1.56x | 2.59 GiB |
| Day 1 (failed boot) | 131072 | — | — | — | 6.96 GiB |

**Available KV is a fixed 4.04 GiB regardless of MML** — it is bounded by weights (15.85 GiB/GPU) + CUDA graphs (0.88 GiB) + the util ceiling, not by MML. MML only sets the window into a fixed pool. (The Day 1 failed boot ran multimodal at util 0.90 and measured 6.96 GiB per-seq at 131072; that per-seq figure is config-independent and anchors the slope.)

### Fitted cost model

```
per-seq KV (per GPU) ~= 1.97 GiB  +  39.2 KiB/token x seq_len
```

- **1.97 GiB floor** = capped SWA layers + base allocation.
- **39.2 KiB/token slope** = the 10 global layers growing linearly.

Validation against all three anchors:

| seq_len | predicted | measured | error |
|---------|-----------|----------|-------|
| 8192 | 2.28 GiB | 2.27 GiB | +0.4% |
| 16384 | 2.58 GiB | 2.59 GiB | -0.4% |
| 131072 | 6.87 GiB | 6.96 GiB | -1.3% |

The two-coefficient sliding-window model holds across a 16x span in length. This is the signature the Week 11 plan predicted: a naive linear projection from the 131072 anchor overshoots short-sequence cost ~8x, because most of the per-token cost at long lengths comes from the globals, which are a minority of layers. The coefficients are architecture-driven — only the available-KV floor (4.04 GiB) sets the pool capacity.

### Chosen working MML

Single-sequence ceiling (text-only, graphs ON, util 0.95): solving `4.04 = 1.97 + 39.2 KiB x L` gives **L ~= 55,400 tokens** of total context (prompt + generation sharing one allocation). Chose **MML 33024** = 32768 top prompt size + 256 generation tail, well under the ceiling.

At MML 33024 the boot reports **max concurrency 1.25x** (cost model predicted ~1.26x — match), per-seq ~3.23 GiB, pool ~41.3K tokens. One full-window sequence fits with ~0.25x of headroom — more comfortable than the 1.01x of a multimodal load at the same MML.

**65536 was dropped:** a 65536+256 sequence needs ~4.43 GiB > 4.04 GiB available — it will not hold even a single sequence in this config (short by ~0.39 GiB). This is a KV-capacity finding, not a tuning miss. The fix (PP, eager mode, or a util bump) was deliberately not applied here to keep the config held-constant for the comparison.

## Single-request throughput sweep (TP=2, c=1)

`tools/throughput_sweep.py` schema v3, `--concurrency 1 --iterations 3 --warmup 1`, prompt sizes `512 2048 4096 8192 16384 32768`, model name passed explicitly. 18/18 requests OK, zero failures.

| Prompt size | Prefill tok/s | Decode tok/s | TTFT (s) | Gen throughput tok/s | Wall (s) |
|------------:|--------------:|-------------:|---------:|---------------------:|---------:|
|   512 | 1952.0 | 44.31 |  0.261 | 42.56 |  6.02 |
|  2048 | 1868.5 | 43.43 |  1.093 | 36.76 |  6.96 |
|  4096 | 1795.0 | 42.23 |  2.276 | 30.79 |  8.31 |
|  8192 | 1689.4 | 40.18 |  4.837 | 22.89 | 11.18 |
| 16384 | 1525.4 | 38.16 | 10.715 | 14.71 | 17.40 |
| 32768 | 1293.2 | 34.88 | 25.278 |  7.86 | 32.59 |

Per-size stdevs were tight throughout (decode sigma < 0.03 tok/s, prefill sigma < 10 tok/s), so the curve is clean, not noisy.

**Decode tok/s vs Gen throughput tok/s** — these measure different things and their divergence is informative:

- **Decode tok/s** is the steady-state generation rate measured over the **decode phase only** (`gen_tokens / decode_time`, excluding prefill/TTFT). It answers "once the model starts emitting tokens, how fast?" and degrades only mildly with context (44.31 -> 34.88, -21%) as global-layer KV attention grows.
- **Gen throughput tok/s** is the **end-to-end** effective rate (`gen_tokens / total_wall_time`, prefill included). It answers "tokens per second the caller actually experiences for the whole request" and collapses hard with context (42.56 -> 7.86, -82%) because prefill (TTFT) increasingly dominates the wall clock.

At 512 the two nearly coincide (44.31 vs 42.56) — prefill is a rounding error on a 6 s request. At 32768 they diverge ~4.4x (34.88 vs 7.86) because 25.3 s of the 32.6 s wall is prefill. The gap between the columns *is* the prefill tax, and it is the headline cost story for long-context single-request serving on this model.

## Cross-checks against Day 1 anchors

**1. Decode @2048 — passes.** 43.43 tok/s vs Day 1 smoke 43.4. Clean tie-in.

**2. Prefill @2048 — passes.** 1868 tok/s vs Day 1 ~1858 (within ~1%, run-to-run).

**3. Prefill "rise then plateau" — reconciled, not a discrepancy.** The cross-check note predicted a rise-then-plateau shape; the actual curve from 512 up is a **monotonic decline** (1952 -> 1293, -34%). These are consistent once the ranges are aligned: the rise the note described was the 128->512 fixed-overhead amortization visible in Day 1's truncated range (1749 -> 1840 -> 1858). By 512 we are already past that knee; from there, O(n^2) prefill attention on the 10 global layers dominates and the rate falls. The "plateau" was the ceiling of Day 1's prompt-size range, not a real plateau. TTFT confirms the quadratic signature — growth per doubling is super-linear and accelerating (~x2.1, x2.2, x2.4 across the top three sizes).

**Decode degradation:** decode eases ~21% (44.31 -> 34.88) as context grows, consistent with the same global-layer KV attention cost climbing with sequence length.

## Validity notes

- `cached_tokens` returned **`null`** on every request (not the `1` the pickup doc anticipated for BOS-only). This appears to be vLLM 0.21.0 not populating the field rather than a cache hit. The nonce prefix is confirmed working independently: per-request `prompt_tokens` jitters by size (509/511/509, 2044/2041/2044, ...), which only happens if each request gets a distinct nonce prefix. No request approached the contamination threshold. Data is trustworthy; flagging the `null` so it isn't misread as a cache miss later.

## Corrections / clarifications logged this session

- **"Top runnable prompt size" -> context-window cap.** The single-sequence ceiling is total sequence length (prompt + generation), not a prompt-size limit. A 32768-token prompt leaves generation headroom under the ~55,400-token ceiling; 32768+256 fits comfortably. Correct framing: achievable context window ~= 55K, top sweep prompt 32768 chosen to sit under it with a generation tail.
- **PP=2 does not halve weights again.** The 15.85 GiB/GPU is already the post-TP=2 figure (~31 GiB FP8 weights, text-only, sharded across the pair). PP=2 partitions by layers across the same 2 GPUs -> same ~50% per GPU, not 25%. Only PP=4 quarters it. Any earlier "PP=2's pool dwarfs TP's" intuition was built on a wrong halving assumption and is retracted. At equal per-GPU weights, PP=2's KV pool should be in the same ballpark as TP=2's; any material divergence would be a second-order (per-stage activation / pipeline buffer / graph memory, or uneven global-layer distribution across stages) effect — and is a Day 3 finding to surface, not prejudge.

## Deferred levers (not applied — would break held-constant config)

- **CUDA-graph memory tradeoff.** Two distinct levers, not interchangeable: `--enforce-eager` reclaims graph memory but kills graph decode (real per-step penalty, would depress every number); `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` reclaims only the profiler's conservative reservation while keeping graph decode (cheaper, risks capture-time OOM). The latter currently can't be tested because `start-vllm.sh` only forwards `HF_TOKEN` — a generic `--env KEY=VAL` passthrough is a pending script change.

(Note: the multimodal/video budget is no longer a deferred lever — text-only is the held-constant baseline config from this point on, baked into `start-vllm.sh`.)

## Deliverables

- Working MML **33024**, available KV **4.04 GiB/GPU**, measured max concurrency **1.25x** at the chosen config (pool ~41.3K tokens) — the per-config "KV pool capacity" entry for the eventual TP-vs-PP table.
- Single-request TP=2 sweep JSON committed to `phase-3-optimization-and-quantization/week-11/results/`.
- `tools/start-vllm.sh` updated to default to text-only (all mm limits 0); committed.
- This journal.

## Status: Day 2 complete

KV characterization done, working MML chosen and validated empirically, c=1 sweep run and cross-checked, text-only baseline locked into the launch script.

## Open threads for Day 3 (no work started)

- **PP=2 on the NVLink pair**, same held-constant config (text-only, graphs ON, util 0.95). Run the same KV characterization first — do not assume the TP pool transfers; compare per-GPU KV pool capacity head-to-head and surface any asymmetry.
- **Concurrent sweeps (c>1).** With the text-only pool (~41.3K tokens at MML 33024, 1.25x), concurrency at large prompt sizes will be KV-capacity-bound, not compute-bound, on this model — the concurrent ceiling is itself a measurement, and is regime-dependent (small prompts are far cheaper per seq than the near-MML figure implies; measure per prompt size).
- Whichever graph/memory config is chosen for the comparison must stay identical across TP=2 (Day 2) and PP=2 (Day 3).
