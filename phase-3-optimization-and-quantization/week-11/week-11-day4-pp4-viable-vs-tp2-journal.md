# Week 11 Day 4 — PP=4 is viable, placement-invariant at c=1, and crosses TP=2 on long-prompt prefill

**Model:** `RedHatAI/gemma-4-31B-it-FP8-block` (FP8 31B Dense, text-only)
**Backend:** vLLM 0.21.0, pinned digest `sha256:a230095847e93bd4...`
**Hardware:** 4× RTX 3090; NVLink pair on physical GPUs 0 & 2 (NV4), all other links host-bridge (PHB)
**Date:** 2026-06-02 (UTC)

## Summary

PP=4 boots cleanly at the matched MML 33024 with a 160,048-token KV pool (4.85× max-concurrency) — roughly 3.9× the TP=2 pool and ~35× the dead PP=2 pool. It is **viable**. Across the matched 512–32768 single-request ladder, the deployer-relevant TP=2 vs PP=4 picture is a genuine tradeoff, not a clean win for either:

- **TP=2 wins decode at every prompt size** (~1.7×; PP=4 decode is 0.55–0.60× of TP=2).
- **TP=2 wins prefill below ~8K; PP=4 wins prefill above ~16K.** The crossover sits between 8K and 16K, near the operating window of the primary workload.
- **Stage→GPU placement is invariant at c=1** for both prefill and decode. Putting the single NVLink hop on the critical path (steered `0,2,1,3`) buys nothing measurable versus the naive all-PHB ordering.

## Where Day 3 left things

PP=2 was closed as non-viable: the un-sharded 256K-vocab embedding/LM-head lands whole on one stage under pipeline parallelism, inflating that stage's weights and crushing its KV floor (1.95 GiB available, vs the 3.22 GiB a single 33024-token sequence needs). PP=2's serviceable context was ~12× smaller than TP=2's. The decision taken at the Day 3 fork was to treat **TP=2 vs PP=4** as the only two practical configs for this model/quant/box and to measure PP=4 — viable *only if* its performance is acceptable, as an empirical posture rather than an assumption.

## Held-constant config

vLLM 0.21.0 (pinned digest), text-only (`--limit-mm-per-prompt` zeroing image/audio/video), CUDA graphs ON, `--gpu-memory-utilization 0.95`, `--kv-cache-dtype auto` (resolves to BF16 KV on Ampere; FP8 KV needs SM 8.9+), `--max-num-batched-tokens 4096`, matched 512–32768 prompt ladder. MML 33024 for all configs so curves overlay the TP=2 Day 2 anchors directly.

## Step 1 — PP=4 KV characterization

Boot at MML 33024 succeeded on the first attempt.

| Metric | Value |
|---|---|
| GPU KV cache size | 160,048 tokens |
| Max concurrency @ 33,024 tok/request | 4.85× |
| Model loading (stage PP0) | 10.02 GiB |

**Per-stage weight asymmetry.** Stage PP0 loaded 10.02 GiB — about 2 GiB heavier than the ~7.9 GiB/GPU predicted from an even weight halving. This is the same mechanism that killed PP=2: the un-sharded input embedding lands whole on the first stage. Resident-memory readings confirm the LM head lands on the last stage (PP3 the heaviest card at ~22.0 GiB, vs ~20.7–20.9 GiB on the others). The two end stages carry the vocab tax; the two middle stages are the lean ~7.9 GiB-class ones. Crucially, four-way splitting gave enough per-stage KV headroom that neither heavy end-stage starved the pool the way PP=2's single fat stage did — PP0 still surfaced 9.36 GiB of available KV.

**Placement (uuid-join, naive boot).** vLLM assigned stage *i* → physical GPU *i*:

| PP rank | Physical GPU | On NVLink pair? |
|---|---|---|
| PP0 | GPU 0 | yes (with GPU 2) |
| PP1 | GPU 1 | no |
| PP2 | GPU 2 | yes (with GPU 0) |
| PP3 | GPU 3 | no |

Under this naive ordering the NVLink pair (GPU 0 & 2) lands on **non-adjacent** stages PP0 and PP2, so the fast link sits entirely off the pipeline critical path — all three stage boundaries (PP0→PP1, PP1→PP2, PP2→PP3) are PHB. This is the worst-case placement, and it was measured first deliberately.

## The decode go/no-go, and a methodology trap worth recording

The real go/no-go was never the pool size — it was whether decode survives three sequential host-bridge hops per token. A single short generation was used as a spot-check before committing to a sweep.

**A first-probe-after-boot artifact nearly produced two wrong conclusions.** The first decode timing on each freshly-booted server consistently came back at ~9.2s for 128 tokens (~13.9 tok/s), while every subsequent probe on the same warm server returned ~5.25s (~24.4 tok/s). Treated as single samples, these produced a "placement is irrelevant" reading, then a "NVLink halves decode latency" reading, then back again — all noise. The resolution came from lining the probes up by ordinality rather than by placement: **the ~9.2s spike is a one-time cold-start cost on the first decode request after boot** (lazy kernel JIT / CUDA-graph replay paths not exercised in warmup), repeatable to ~the same magnitude on both placements, and independent of stage ordering.

Lesson recorded: n=1 timing probes on this host-staged path are unreliable; the variable that mattered was warm-vs-cold, not placement. This is the empirical justification for the sweep harness's `--warmup 1` — that discarded warmup wave absorbs exactly this cold-start so it never contaminates a measured iteration.

A second methodology fix surfaced here too: greedy decode (`temperature 0.0`) can hit EOS early and generate far fewer than `max_tokens`, making wall-clock uncomparable. All comparison probes and the sweep were run with a fixed generation length to hold decode work constant.

**Verdict: GO.** Warm decode is ~24 tok/s, tolerable for single-user streaming, slower than TP=2 but not in the "x1 has destroyed decode" floor that would have killed the config. PP=4 does not join PP=2 in the dead column.

## Step 2 — Single-request sweep (c=1), TP=2 vs PP=4, both placements

Three c=1 sweeps were run (3 measured iterations + 1 warmup per prompt size, nonce-prefixed prompts to defeat prefix caching, fixed 256-token generations). Decode stdev across iterations was in the thousandths of a tok/s — these are trustworthy curves, not noisy probes.

### Decode (tok/s) — TP=2 wins everywhere; placement-invariant

| prompt | TP=2 | PP=4 naive | PP=4 steered | PP=4 / TP=2 |
|---:|---:|---:|---:|---:|
| 512 | 44.31 | 24.18 | 24.18 | 0.55 |
| 2048 | 43.43 | 23.71 | 23.70 | 0.55 |
| 4096 | 42.23 | 23.65 | 23.65 | 0.56 |
| 8192 | 40.18 | 23.01 | 23.01 | 0.57 |
| 16384 | 38.16 | 22.32 | 22.31 | 0.59 |
| 32768 | 34.88 | 21.06 | 21.06 | 0.60 |

Naive and steered decode are identical to three significant figures at every size. The NVLink hop does nothing for decode because the per-token inter-stage payload (one token's hidden state, tens of KB) is far too small for bandwidth to matter — decode is latency-bound, and the per-hop latency floor is the same handshake on either link.

### Prefill (tok/s) and TTFT — the crossover

| prompt | TP=2 prefill | PP=4 steered prefill | TP=2 TTFT (s) | PP=4 steered TTFT (s) |
|---:|---:|---:|---:|---:|
| 512 | 1952.0 | 904.3 | 0.261 | 0.563 |
| 2048 | 1868.5 | 948.4 | 1.093 | 2.153 |
| 4096 | 1795.0 | 924.0 | 2.276 | 4.422 |
| 8192 | 1689.4 | 1360.2 | 4.837 | 6.007 |
| 16384 | 1525.4 | 1728.2 | 10.715 | 9.458 |
| 32768 | 1293.2 | 1812.1 | 25.278 | 18.041 |

TP=2 prefill **declines** with prompt length (1952 → 1293); PP=4 prefill **rises** (904 → 1812). They cross between 8K and 16K. By 32768, PP=4 prefill is ~1.4× TP=2's and PP=4's TTFT is ~7s lower (18.0s vs 25.3s). Naive PP=4 prefill tracks steered within ~1–3% with no systematic direction (naive faster at 512, steered faster at 32768), confirming prefill is placement-invariant at c=1 as well.

**Mechanism.** TP splits every matmul and pays a per-layer all-reduce whose cost scales with activation size, so its prefill throughput erodes as the prompt grows. PP has no per-layer collective — it ships activations across each stage boundary once per stage transition. At short prompts PP is dominated by the pipeline fill bubble (hence the poor 512 number); as the prompt grows the bubble amortizes and PP's lack of a length-scaling communication tax lets prefill scale up while TP's scales down. The crossover is the structural consequence.

## Conclusions

1. **PP=4 is viable.** Boots at matched MML 33024, 160K-token pool (~3.9× TP=2), no stage starved.
2. **Neither config dominates at c=1.** TP=2 is the decode-throughput and short-prompt-prefill choice; PP=4 is the long-prompt-prefill / lower-long-TTFT / larger-KV-pool choice. The decode penalty (~1.7×) is the price of pipelining over a host-bridge fabric.
3. **Stage placement is irrelevant at c=1.** Topology-aware ordering (NVLink hop on the critical path) does not measurably improve prefill or decode. The decode cost is structural to 4-stage pipelining, not to which specific hops are slow — it cannot be optimized away by placement at single-request load.
4. The earlier ~9.2s "placement effect" was a cold-start artifact, conclusively ruled out by two tight low-variance sweeps per placement.

## Provenance notes

- The naive and original steered c=1 sweeps were produced before the harness gained a `--placement` tag; the original steered 512 prefill (780 tok/s) was a small-prompt prefill outlier and was superseded by a re-run with the tagged harness (904 tok/s, consistent with naive's 854). Decode was identical across all three runs.
- The re-run steered file carries `sweep_config.placement = "steered"` and a placement-tagged filename; the naive file carries `"naive"`. The harness `placement` field is provenance only and is not self-verified — actual placement was confirmed each boot via `nvidia-smi` uuid-join.
- Tokenizer `nonce_tokens` calibration varied across runs (7–12 tokens); this is swamped by the prompt sizes and does not affect the comparison.
- `cached_tokens` reported `null` throughout on vLLM 0.21.0 (field not populated, not a cache hit); contamination control relied on per-request `prompt_tokens` jitter from the nonce prefix, which held.

## Deliverables committed

- PP=4 KV characterization and per-stage asymmetry (above).
- Placement record (naive and steered), uuid-confirmed.
- Go/no-go verdict: **viable**, with decode evidence.
- c=1 sweep JSON for TP=2, PP=4 naive, and PP=4 steered, under `phase-3-optimization-and-quantization/week-11/results/`.
- Harness change: `--placement {naive,steered,na}` added to `tools/throughput_sweep.py` (additive; schema v3 unchanged), folded into the default output filename when not `na` and recorded in `metadata.sweep_config.placement`.
- Launcher change: `--device-order` added to `tools/start-vllm.sh` for deterministic stage placement via in-container `CUDA_VISIBLE_DEVICES` + `CUDA_DEVICE_ORDER=PCI_BUS_ID`.

## Forward look (planned, not started)

- **Day 5 — concurrency (c>1) sweeps** at the held-constant MML 33024, TP=2 and PP=4, matched 512–32768 ladder. Open question: whether placement-invariance survives at c>1. Unlike single-request decode, concurrent waves keep multiple requests' activations crossing the inter-stage links simultaneously — the first regime where NVLink bandwidth on the critical path could plausibly matter. If PP=4 concurrency turns out placement-sensitive, both placements get swept. Achievable concurrency is KV-regime-dependent and must be measured per prompt size, not extrapolated from the c=1 cost model.
- **Day 6 (new) — maximum context window.** Push MML toward each config's hardware ceiling and rerun both c=1 and c>1 sweeps at the stretched window. Motivated by expanded context-window requirements in the primary workload. MML and concurrency draw on the same KV budget, so this is a distinct question from Day 5 (max window vs. batch capacity at a fixed window) and is kept as a separate day to avoid confounding the two variables. Expected to establish the max serviceable context per config — bounded by the heaviest stage (the embedding/LM-head end stages) under PP — and the concurrency-at-max-context tradeoff.
