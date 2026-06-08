# Week 11, Day 5 — TP=2 wins under load: the c=4 concurrency matrix

**Date:** 2026-06-07/08
**Model:** `RedHatAI/gemma-4-31B-it-FP8-block` (31B Dense, FP8 weights)
**vLLM:** `vllm/vllm-openai:v0.21.0`, digest `sha256:a230095847e93bd4df9888b33dab956fa9504537b828a23657d2b26fed57b5c9` (verified via `RepoDigests` pre-session)
**Git SHA:** `6859ac6` (clean tree — `tools/` changes from Day 4 committed before any sweep; all three result files record `dirty:false`)
**Held-constant config:** text-only, CUDA graphs ON, `--gpu-memory-utilization 0.95`, `--kv-cache-dtype auto`, `--max-num-batched-tokens 4096`, MML 33024, prompt ladder 512–32768, `max_tokens=256`, `--iterations 3 --warmup 1`

---

## TL;DR

At concurrency 4, **TP=2 beats PP=4 on aggregate generation throughput at every prompt size**, and on fan-out completion time at every prompt size — including the long-context end where TP=2's KV pool physically cannot hold four full-window sequences (it serializes them and still wins). PP=4's much larger KV pool is a real *capacity* advantage — it keeps ~4× as many long-context sessions resident — but at this concurrency that capacity does not convert into throughput, and for synchronized fan-out it actively hurts. The one case PP=4 wins is the prefill of a *single* long-context request. Stage placement showed its first-ever systematic signal under load (steered ahead in all 18 waves) but at sub-1% it remains deployment-irrelevant.

---

## The question Day 5 set out to answer

Day 4 settled the c=1 (single-request) picture: PP=4 is viable, with a 160,048-token KV pool (max concurrency 4.85× at MML 33024 — roughly 3.9× TP=2's pool); TP=2 wins decode everywhere (~1.7×); PP=4 wins prefill above ~16K; and which GPU each pipeline stage lands on doesn't matter at c=1.

Day 5 tested a specific thesis: that the c=1 comparison **understates PP=4**. PP=4 can hold ~4 full-window sequences resident at once where TP=2 can barely hold one, so under concurrent load its larger pool *should* let it serve more sessions and win on aggregate system throughput, even though each individual request decodes slower.

**The thesis was refuted.** The extra capacity is real but it doesn't pay off as throughput at c=4. The rest of this entry is the evidence and the why.

### Three terms used throughout

- **Concurrency / c=4** — the benchmark client fires 4 requests at the same instant, waits for all 4 to finish, then fires the next group of 4. Each group of 4 is a "wave." This is a deliberate stand-in for a burst of simultaneous requests.
- **Aggregate generation throughput** — a wave's total output tokens divided by the wave's wall-clock span. This is the right "how much total work does the box do" metric. It is *not* the same as "how fast does a burst of requests come back" — that question is answered by the latency tables further down, and the two answers can point in different directions.
- **Fan-out** — the agentic pattern this work is ultimately aimed at. An *orchestrator* agent decomposes a task and distributes the pieces (e.g. "analyze this component's logs") to several *sub-agents* that run at the same time, then collects their findings. A fan-out of N sub-agents dispatched together is exactly a wave of N concurrent requests — which is why the wave results below map directly onto that use case. A fan-out completes only when its *slowest* sub-agent returns.

## Method

Three sweeps at `--concurrency 4` on the identical 512–32768 prompt ladder, changing exactly one thing per server boot:

1. **PP=4 steered** (`--device-order 0,2,1,3`; the one NVLink link sits on the PP0→PP1 stage boundary) — does PP=4 batch well under load?
2. **PP=4 naive** (pipeline stage *i* → physical GPU *i*; all three stage boundaries on slow PCIe/PHB links) — the placement test: does the NVLink link matter now that it didn't at c=1?
3. **TP=2** (GPUs 0,2 — the NVLink pair) — the deployment comparator.

Framing decision made at session start and held: run the straight c=4 ladder on all three configs on identical axes and let TP=2 fail honestly at long context, rather than hunting per-size for each config's maximum sustainable concurrency. (TP=2 did not fail — it serialized; see Finding 3.)

Stage placement was confirmed on every boot via `nvidia-smi` UUID-join, never trusted from the launcher's intent line. PP=4's KV pool line reproduced Day 4 exactly on both boots (160,048 tokens / 4.85×). All waves passed the nonce-integrity check (per-request `prompt_tokens` jitter present, no prefix-cache contamination warnings).

**Result files** (committed to `phase-3-optimization-and-quantization/week-11/results/`):
- `throughput_sweep_vllm-openai_gemma-4-31B-it-FP8-block_c4_steered_20260607T225838Z.json`
- `throughput_sweep_vllm-openai_gemma-4-31B-it-FP8-block_c4_naive_20260607T233312Z.json`
- `throughput_sweep_vllm-openai_gemma-4-31B-it-FP8-block_c4_20260608T000207Z.json` (TP=2, placement=na)

Schema v3 records per-request `prefill_time_s`, `decode_time_s`, `ttft_s`, and dispatch/completion timestamps, which is what made the prefill-under-load analysis (Finding 5) possible without any new runs.

## Headline table — aggregate generation throughput at c=4

| prompt | TP=2 (tok/s) | PP=4 steered (tok/s) | PP=4 naive (tok/s) | TP=2 / PP=4 |
|-------:|-------------:|---------------------:|-------------------:|------------:|
|    512 |        140.5 |                 77.0 |               76.9 |       1.82× |
|   2048 |         93.2 |                 50.5 |               50.3 |       1.85× |
|   4096 |         64.1 |                 34.5 |               34.4 |       1.86× |
|   8192 |         31.6 |                 22.1 |               22.0 |       1.43× |
|  16384 |         18.1 |                 12.2 |               12.1 |       1.48× |
|  32768 |          7.9 |                  5.8 |                5.7 |       1.36× |

Wave-to-wave repeatability was tight on all three configs (≤ ~1 tok/s across the 3 iterations at every rung). Zero failed requests anywhere — but the TP=2 32768 row's `4/4 ok` is misleading without the serialization context in Finding 3.

## Finding 1 — PP=4's batching gain shrinks as prompts grow, and inverts above 16K

How much faster does PP=4 go at c=4 than it did running one request at a time (its own Day 4 c=1 aggregate anchors)? Ideal batching would be 4×.

| prompt | c=1 agg | c=4 agg | speed-up |
|-------:|--------:|--------:|---------:|
|    512 |   23.05 |    77.0 |    3.34× |
|   2048 |   19.83 |    50.5 |    2.55× |
|   4096 |   16.84 |    34.5 |    2.05× |
|   8192 |   14.98 |    22.1 |    1.48× |
|  16384 |   12.26 |    12.2 |    1.00× |
|  32768 |    8.49 |     5.8 |    0.68× |

At 512 tokens PP=4 batches well (3.34× of its single-request rate). The benefit then decays steadily, reaches **exact break-even at 16384**, and **goes negative at 32768**: firing four full-window requests at once was 32% *slower* than running the same four one after another.

This is **not** the memory wall. The 160K pool held all four sequences as designed — zero failures, zero preemptions. It is the **compute/scheduler wall**. With `--max-num-batched-tokens 4096`, a 32K-token prefill takes 8 scheduler rounds to chew through. Four of those prefills compete for the same per-round token budget and stretch into a long prefill train, and vLLM's chunked-prefill interleaving makes the already-running requests share each round with the still-prefilling ones — so every in-flight decode slows down. The per-request decode spread is the fingerprint: a tight 22.5–22.5 tok/s at 512 widening to 1.6–17.2 tok/s at 32768. The first request to finish prefilling decodes near full speed; the last spends most of the wave waiting to start.

**Takeaway:** PP=4's pool buys the *capacity* to hold four long sequences, but at long context under this token budget that capacity is worth less than nothing for throughput. The batching payoff lives entirely below ~8K.

*Caveat:* `--max-num-batched-tokens 4096` directly shapes this trade-off — a bigger budget would clear the prefill train faster at the cost of worse decode starvation for in-flight requests. It is held constant this week and was not varied; the inversion is a property of *this config*, not a universal law. Varying that knob is a clean future experiment with a direct line to the agentic workload (see Implications).

## Finding 2 — first systematic placement signal, still operationally irrelevant

At c=1 (Day 4), steered vs naive were identical to three significant figures with no consistent direction. At c=4 a direction finally emerged: **steered beat naive at all six prompt sizes, in all 18 waves**, by ~0.3–1% on aggregate throughput and correspondingly on wall-clock (e.g. 177.8s vs 178.7s at 32768).

This matches the predicted mechanism. At c=1 each stage boundary carries one token's hidden state per step — tiny, latency-bound, bandwidth-irrelevant, which is exactly why the fast NVLink link didn't help. At c=4 the boundaries carry activations for multiple in-flight requests, the per-crossing payload grows, and the NVLink link on one of three boundaries starts to register. The mechanism is confirmed; the magnitude settles the practical question.

**Verdict:** placement-invariance effectively extends to c=4. The consistent sub-1% steered edge is the first systematic placement signal in this program, and worth recording as confirmation of the mechanism — but it is far below deployment significance. Topology-aware stage placement remains a non-factor at this concurrency. (Whether the gap widens at higher concurrency is untested — Step 4 was deferred.) The Finding 1 inversion is identical across placements, confirming it is a scheduler phenomenon, not a topology one.

## Finding 3 — TP=2 doesn't fail at long context; it serializes (and still wins)

The expected failure at the top of the ladder — TP=2's ~41K-token pool cannot hold 4 × 33K sequences — did not show up as errors. **vLLM queued instead of rejecting.**

The diagnostic is the decode rate at 32768: a uniform **34.9–34.9 tok/s, exactly the c=1 decode anchor (34.88)**. The scheduler ran the four requests effectively one at a time, each at full single-request speed. The arithmetic confirms it: a c=1 request at 32K is ~25.3s prefill + ~7.3s decode ≈ 32.6s; ×4 ≈ 130.5s predicted vs **130.3s observed wall**. At 16384 the decode spread (14.6–36.9) shows partial concurrency — roughly 2–3 sequences resident at a time, the rest queued.

**Recording discipline:** the TP=2 32768 rung must be read as **"effective concurrency ≈ 1 (scheduler-serialized)"**, not as a genuine c=4 batching result. The `4/4 ok` is truthful but misleading without this label. The graceful-degradation behavior (queue, don't reject) is itself a useful deployment property — see Finding 4 for what happens when even queueing can't keep up.

Where its pool allows, TP=2 batches excellently: 3.3× of its own single-request rate at 512 (140.5 vs ~42.6), and its *worst* decode rate at 512 (36.6 tok/s) beats PP=4's *best* (22.5 tok/s).

## Finding 4 — capacity: why PP=4's pool is 3.86× bigger, and why that's separate from throughput

The TP=2 boot log lets us anchor the capacity story to the cost model rather than just asserting it:
- TP=2: available KV 4.04 GiB/GPU → **41,427-token pool, max concurrency 1.25×** (= 1.25 full windows: one sequence plus a sliver).
- PP=4: **160,048-token pool, 4.85×**.
- Ratio: 160048 / 41427 = **3.86×** full-window sequences resident. That single number *is* PP=4's capacity advantage.

**Cost-model confirmation.** At max concurrency 1.25, TP=2's 4.04 GiB of KV holds 1.25 sequences → 3.23 GiB per full-window sequence per GPU. The Day 2 model predicts 1.97 GiB + 39.2 KiB/token × 33024 = 3.20 GiB — within ~0.8% of the log. The measured pool validates the analytical model again; we can trust it for Day 6 MML reasoning.

**Where the 3.86× comes from — two compounding effects:**
1. **GPU count (the dominant factor).** PP=4 spans four cards, TP=2 spans two — roughly twice the post-weights VRAM pooled into KV. Strategically, PP=4 recruits GPUs 1 and 3 — the PCIe x1 cards that are *useless* for tensor parallelism, because the x1 link strangles TP's per-token all-reduce collectives. Pipeline parallelism passes only one hidden-state vector across each stage boundary per token, which the x1 link handles fine on memory-bandwidth-bound decode. So PP=4 monetizes two otherwise-dead GPUs as KV storage. **TP=2 structurally cannot reach that capacity** — it's a hardware-topology limit, not a tuning gap.
2. **Per-stage KV cost (the secondary factor).** Each PP stage stores KV only for its ~quarter of the layers, so a token costs each stage roughly a quarter of the full-model per-token KV. This is why the advantage exceeds a clean 2×-for-2×-GPUs. It doesn't reach a clean 4× because the un-sharded embedding (whole on PP0) and LM head (whole on PP3) eat into the end stages' room, and the pool is bounded by the *tightest* stage — the same asymmetry that, with only two stages to absorb it, killed PP=2 on Day 4.

**What happens past the pool limit (the conditional on the capacity claim).** Neither pool is a hard admission cap — vLLM admits more requests than fit and then keeps the resident set within the pool by *preempting*: it evicts a running sequence's KV and, by default in v1, *recomputes* it from scratch when the sequence is rescheduled (the alternative, swapping KV to CPU RAM, would be punishing here because the PP=4 stages on GPUs 1 and 3 sit behind PCIe x1). So over-subscription doesn't error — it degrades, as throughput collapse and latency-tail blow-up while the scheduler thrashes sequences in and out, paying repeated re-prefill. The clean serialization at TP=2/32K (Finding 3) is the *gentle* end of this behavior — four sequences taking neat turns; the ugly end is many long sequences evicting each other mid-decode. The practical reading: PP=4's 3.86× doesn't remove the wall, it moves *where* the wall sits — both configs thrash once offered load exceeds their pool, PP=4 just tolerates ~3.86× more concurrent long-context sessions first. (One genuine hard limit underneath the soft one: a single request whose context exceeds the *whole* pool cannot run at all — which is why MML can't be set above what one sequence's KV fits in. That's the Day 6 question.)

**A v0.21.0 wrinkle worth knowing for Day 6.** The TP=2 boot log flags that CUDA-graph memory profiling (default since v0.21.0) reserves capture memory *before* KV, so `--gpu-memory-utilization=0.95` is effectively 0.9093 for KV purposes; recovering the old KV size needs ~0.9907, or `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`. This does **not** affect today's comparison — graphs are ON in the held-constant config, so both TP=2 and PP=4 pay it on all their cards, and the 3.86× ratio stands. But it means both absolute pools are ~4 utilization points smaller than a naive reading of 0.95 implies, i.e. there is recoverable KV on the table. That's a **Day 6 lever** (max-MML day), not a retroactive change here — touching it now would alter decode behavior and break the c=4 comparison just closed.

**The conceptual point not to conflate:** capacity and throughput are different ceilings, and today proved they don't move together. PP=4's 3.86× is a genuine *resident-sequence* advantage — it keeps ~4 long sessions warm where TP=2 keeps one and queues the rest, which buys multi-turn interactivity and MML headroom. It does *not* buy aggregate throughput at c=4. For a fan-out, capacity tells you how many sub-agents can be *resident*; it says nothing about how fast their findings come back.

## Finding 5 — prefill under load: PP=4 wins one request, TP=2 wins the fan-out

Because schema v3 stores per-request `prefill_time_s` separately, the prefill question can be answered from today's JSONs with no new runs. It needs two readings, because "fastest prefill per request" and "fastest fan-out completion" are different questions with different answers.

**Reading A — the first request in each wave to start prefilling.** When the client fires four requests at once, they compete for the scheduler; one wins the first slot and the other three queue behind it. That first request's prefill time is the least distorted by queueing, so it's the fairest estimate of what a prefill actually costs under load:

| prompt | TP=2 (s) | PP=4 steered (s) |
|-------:|---------:|-----------------:|
|    512 |     0.29 |             1.99 |
|   2048 |     1.12 |             4.21 |
|   4096 |     2.33 |             5.09 |
|   8192 |     4.89 |             6.01 |
|  16384 |    10.76 |             9.47 |
|  32768 |    25.37 |            18.13 |

**The c=1 prefill crossover reproduces under load.** Crossover still between 8K and 16K; PP=4's 32K number (18.13s) matches its Day 4 c=1 prefill TTFT (18.0s), and TP=2's (25.37s) matches its (25.3s). PP=4's long-context per-request prefill advantage is real and survives concurrency.

**Reading B — the last request in each wave to finish.** A fan-out is only done when its slowest member returns, so this is the number an orchestrator actually waits on. Two versions: prefill-only (relevant when a sub-agent's output is short) and full 256-token output:

| prompt | prefill-only TP=2 (s) | prefill-only PP=4st (s) | 256-output TP=2 (s) | 256-output PP=4st (s) |
|-------:|----------------------:|------------------------:|--------------------:|----------------------:|
|  16384 |                 49.67 |                   70.57 |               56.57 |                 84.06 |
|  32768 |                123.02 |                  162.95 |              130.33 |                177.81 |

**TP=2 wins fan-out completion at every prompt size, for both short-output and 256-output workloads** — even at 32K where its individual prefill is *slower*. The reason is Finding 3: TP=2 serializes cleanly, running four prefills back-to-back at full single-stream rate with no interleaving drag, so its last request finishes at 123s. PP=4's chunked-prefill interleaving spreads the four prefills across the whole wave, so *its* last request finishes at 163s despite each individual prefill being faster.

**Synthesis:** PP=4 wins exactly one scenario — the prefill of a *single* long-context request with no fan-out. The moment work fans out, TP=2's clean serialization beats PP=4's interleaved pipeline on completion time, whether the output is short (prefill-dominated) or 256 tokens. The prefill reread therefore *strengthens* the deployment verdict for fan-out workloads while preserving one honest nuance for the lone-long-request case.

*Caveat on Reading B:* per-request times at c>1 include a client-side observation artifact (the requests in a wave share one event loop). The last-to-finish and wall-clock numbers are the trustworthy system-level metrics; the first-to-start number (Reading A) is the least-distorted per-request proxy. Both readings are internally consistent and consistent with the c=1 anchors, so the conclusions hold.

## The Day 5 verdict — scoped

**TP=2 wins under load at this MML, on every metric a fan-out workload cares about:** aggregate generation throughput (every prompt size), and fan-out completion time (every prompt size, short-output or 256-output). The sharpest single result: at 32768 tokens, TP=2 running four requests *sequentially* beat PP=4 running them *concurrently* by 36% on throughput and finished the fan-out ~47s sooner.

**PP=4 wins exactly one thing on speed:** the prefill of a single long-context request (18.1s vs 25.4s at 32K), which reproduces the c=1 crossover. It also wins on **capacity** — 3.86× the resident long-context sessions — which is a real advantage for multi-turn interactivity and MML headroom, just not for throughput or fan-out latency at c=4.

Choosing between them is a workload-shape decision, not a throughput decision: PP=4 if you need many long sessions resident and responsive turn-by-turn; TP=2 if you need total work done fast or fan-outs returned fast. On both of the latter, the answer is unambiguous.

## Incidental finding — no FlashAttention anywhere this week

Checked mid-session: vLLM **forces `TRITON_ATTN` for Gemma 4** —

> `Gemma4 model has heterogeneous head dimensions (head_dim=256, global_head_dim=512). Forcing TRITON_ATTN backend to prevent mixed-backend numerical divergence.`

So no FlashAttention-class kernel ran in any Week 11 experiment. The same heterogeneous head-dimension property that made the KV cost model two-coefficient (Day 2) also dictates the attention backend. FlashInfer appears only for top-p/top-k sampling. Two implications: (1) the backend is forced by the architecture, so it was identical across TP=2 / PP=2 / PP=4 — an implicit held-constant, now documented; (2) there may be attention-kernel headroom this model can't access on this stack, relevant when comparing these absolute numbers to other models or stacks.

## Implications for the agentic workload (forward-looking, not measured today)

The work is broadening from the narrow statmon-ai case toward my primary use case: an orchestrator agent that spawns ephemeral sub-agents — e.g. one per component to analyze that component's logs and report findings back. This is the fan-out pattern defined above.

- **Sub-agent fan-out is exactly the wave pattern measured here**, so today's results are unusually representative of this workload.
- **Log analysis is long-prompt, prefill-dominated work** — the top of the ladder, the inversion/serialization zone. Two concrete facts now in hand: (1) at long context, firing the fan-out in parallel is *not* faster than dispatching sequentially — and TP=2's scheduler does the sequential thing automatically; (2) TP=2 returns the whole fan-out sooner than PP=4 at every size (Finding 5B).
- **Prefix caching helps less than in the statmon-ai case.** Sub-agent prompts invert the static/dynamic ratio: a small cacheable role prompt plus a large unique log payload. Mitigation: order prompts shared-base → role suffix → logs so the common stem caches; anything varying early invalidates everything after it.
- **The orchestrator blocks on the slowest sub-agent** — the last request in the wave to finish, which interleaving makes ugly (down to 1.6 tok/s decode for PP=4 at 32K). Per-request tail latency, not aggregate throughput, is the orchestrator's experienced cost — and Finding 5B measures it directly.
- **Design levers, in payoff order:** (1) scope sub-agent inputs — a pre-filtered log excerpt moves the workload from the 16–32K inversion zone down to 2–8K where batching pays 1.5–2.5× and the latency spread is tolerable, and (because each sequence is smaller) lets the same pool hold far more sub-agents resident before the preemption thrash of Finding 4 sets in; (2) make orchestrator dispatch serving-aware — bounded fan-out at moderate context, sequential at long context; (3) `--max-num-batched-tokens` as a future tuning experiment (the knob behind the Finding 1 inversion).

## Methodology notes

- **`dirty:true` prevented.** `tools/` committed (`6859ac6`) before any sweep; all three result files record a clean SHA.
- **`--warmup 1` retained** on every sweep; no cold-start spikes in measured waves; the Day 4 cold-start artifact did not recur.
- **Placement verified by UUID-join on every boot.** Steered: rank order on GPUs 0,2,1,3 (PP1 on the NVLink peer). Naive: stage *i* → GPU *i*. TP=2: both ranks on the NVLink pair (0,2), symmetric ~23.4 GiB footprints.
- **Closed-wave caveat (applies to all c>1 numbers):** the dispatch-4 / drain / repeat pattern pays a synchronized prefill storm at wave start and runs a thinning batch at the tail; true steady-state continuous-arrival throughput at the same concurrency would be somewhat higher. The c=4-vs-c=1 comparisons are internally valid (both closed-loop on the same harness), so the inversion and the TP-vs-PP ordering survive the caveat; absolute steady-state throughput does not follow from these numbers.
- **Tokenizer calibration drift:** nonce_tokens calibrated to 11 on the PP boots, 9 on the TP boot — run-to-run jitter in the same procedure; immaterial at these prompt sizes, recorded for completeness.

## Open questions carried forward

1. **Step 4 (deferred):** higher-concurrency probe at 512–4096 — where does each config's aggregate peak (c=8? c=16?), and where is the scheduler/compute wall?
2. **Does the sub-1% placement gap widen at higher concurrency?** Untested; only worth revisiting if Step 4 happens.
3. **`--max-num-batched-tokens` sensitivity** — the knob behind the Finding 1 inversion. Separate, labelled experiment; breaks held-constant if mixed into Week 11.
4. **CUDA-graph KV tax recovery** (Finding 4) — quantify KV recovered at util 0.9907 or with the profiler env var; a Day 6 input.

## Day 6 (planned)

Maximum-MML characterization — how far the context window stretches per config, motivated by the expanded context requirements of my primary use case. MML was held at 33024 today per plan; Day 6 is the day it moves, and the cost-model confirmation (Finding 4) plus the CUDA-graph KV tax are the two inputs that feed it.
