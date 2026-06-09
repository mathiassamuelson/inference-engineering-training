# Week 11, Day 6 — Maximum context length: TP=2 interactive ceiling vs PP=4 256K, and the CUDA-graph KV tax

**Date:** 2026-06-08 / 09
**Config under study:** `RedHatAI/gemma-4-31B-it-FP8-block`, vLLM 0.21.0 (pinned digest `sha256:a230095847e93bd4…`), text-only, 4×RTX 3090, NVLink pair on GPUs 0+2.
**Question:** How far does the context window stretch on this box for TP=2 and PP=4, how much context does recovering the CUDA-graph KV tax buy, and — the part that actually matters for the deployment — is the long context *usable* or just *bootable*?

## Short version

- **TP=2 baseline ceiling (util 0.95): 54,496 tokens.** KV-bound. Decode ~33 tok/s at the ceiling.
- **TP=2 recovered ceiling (util 0.97): 66,848 tokens** — about +12,350 tokens (+22.7%) over baseline, and it crosses the 64K context tier that was out of reach at 0.95. Still KV-bound, still ~32 tok/s, TTFT in the seconds. This is the **interactive** ceiling.
- **PP=4 ceiling (util 0.95): 262,144 tokens (256K)** — the model's full architectural max. **Architecture-bound, not KV-bound:** there was KV to spare at the wall (max-concurrency 1.30× at 256K). But it serves the long-context end at ~15 tok/s with a multi-minute time-to-first-token, which is not usable for an operator in the loop.
- **The two CUDA-graph tax-recovery recipes the vLLM boot log recommends both failed on this box.** Disabling the estimate OOMs at sampler warmup; util 0.9907 is rejected at init for insufficient free memory. The only working recovery path was an intermediate util (0.97) found by laddering.
- **Deployment read (corrected mid-session):** neither single config serves the primary use case well. TP=2 is interactive but context-limited; PP=4 has the context but is not interactive. The long-context need is met by *architecture* — an orchestrator on 31B TP=2 at its interactive ceiling, delegating bulk-context work to smaller sub-agents — not by maximizing context on one config. Today's PP=4 measurement is the evidence that single-config context-maximization does not produce an interactive system.

## Setup notes

The pre-session check caught an untracked scratch file (`tools/extract-fields.py`) before any committed artifact ran. Removed it; tree clean before the first boot. This is the same check that prevented dirty-SHA results on Day 4 — worth keeping at the top of every session.

One tool change today: added a `--profiler-cudagraphs {on,off}` flag to `tools/start-vllm.sh`. `on` is the default and preserves the Days 1–5 baseline behavior (CUDA-graph memory estimate enabled). `off` injects `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`. The flag is opt-in by design — the tax-recovered condition changes held-constant, so it must never be a silent default. Committed the launcher change *before* any results-writing probe, so all result JSONs record a clean SHA.

## TP=2: a KV-bound ceiling

Stepped MML up at util 0.95. The KV pool is **not** constant across MML — it grows. Understanding why is worth the space, because it's the same effect that dominates the PP=4 result and it governs how the whole ladder has to be read.

Start with what the pool number is. vLLM reserves everything it *must* hold (model weights, CUDA-graph memory, overhead, and the prefill working set — the scratch memory to run a forward pass), then hands the leftover to the KV cache. So the pool is a division:

```
pool (token-slots that fit)  =  available KV memory  /  per-token KV cost
```

To know which way the pool moves as MML changes, watch both halves of that fraction.

*The bottom (per-token cost) falls as MML rises.* Gemma 4 is a hybrid-attention model — sliding-window (SWA) layers plus global layers. The SWA layers only cache a fixed window of recent tokens, so once a sequence is longer than that window their cost stops growing. Spread that fixed window cost over a longer sequence and the *average* per-token cost drifts down toward the global-layer floor. (Full derivation in the PP=4 section.) A smaller denominator means more token-slots fit.

*The top (available KV memory) stays roughly constant as MML rises — and that's chunked prefill's doing.* The prefill working set is sized by the largest number of tokens processed in a single forward pass. Chunked prefill (`--max-num-batched-tokens 4096`) caps that at 4,096 tokens regardless of how long the prompt or MML is, so the working-set reserve is pinned and the leftover for KV doesn't move as we climb.

Constant top, falling bottom → the pool rises smoothly, every rung. That's the clean 46,965 → 51,765 → 53,883 climb below.

The counterfactual is what makes chunked prefill's role clear. If prefill were *not* chunked, a full MML-length prompt would be processed in one pass, so the working set would have to be sized for MML tokens — and would grow every time we raised MML. Now both halves of the fraction move: the falling per-token cost pushes the pool up, the swelling working set pushes it down, and they fight. The pool might rise, go flat, or shrink, and a failed high rung would be ambiguous — out of KV capacity, or just couldn't fit the working set? Chunked prefill holds the top still so the amortization shows through as a clean, single-cause trend. It didn't make the pool *rise* (amortization did that); it made the ladder *readable*.

The practical consequence: because the pool isn't a single fixed number, the ceiling can't be read off one boot — you have to walk it.

```
MML 40960:  pool 46,965 tok   max-conc 1.15x
MML 49152:  pool 51,765 tok   max-conc 1.05x
MML 53248:  pool 53,883 tok   max-conc 1.01x
MML 55296:  FAILS — needs 4.07 GiB/seq > 4.04 GiB available
            engine's own estimate of the ceiling: 54,496 tokens
```

vLLM's pre-flight KV check solved the ceiling for us: at 55296 it refuses to boot and reports the largest MML that fits one full window — **54,496 tokens**. Booting at exactly 54,496 came up clean at max-concurrency 1.00× (one full window fits, nothing more).

Placement confirmed by UUID-join: both ranks resident on physical GPUs 0 and 2 (~23.3 GiB each), x1 cards at 1 MiB. Steered correctly.

**Functional probe (the real test).** A clean startup line is necessary but not sufficient. Probed at ~90% of the ceiling (49K-token prompt, concurrency 1):

```
prompt 48,886 tok → decode 33.4 / 33.3 tok/s, prefill ~1,130 tok/s, completes clean
```

Ceiling confirmed usable. **Cost-model check:** predicted ~55,300, measured 54,496 → **−1.5%** (the model was slightly optimistic because it assumed a fixed KV budget; the working set grows with MML). Good agreement.

## The CUDA-graph KV tax: real, partially recoverable, and the boot log's advice doesn't survive contact with this box

Since vLLM 0.21.0, CUDA-graph memory profiling reserves graph-capture memory *before* KV. At util 0.95 the boot log reports that ~0.96 GiB (~13K tokens of KV) is held back this way, and suggests two recovery paths. **Both failed here.**

**Recipe 1 — disable the estimate (`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`), keep util 0.95.** KV allocation succeeded and *advertised* a bigger pool (5.0 GiB / 67,444 tokens, max-conc 1.24× at MML 54496). But the server then **OOM'd at sampler warmup** before it ever came up. The estimate isn't a free tax — it's reserving memory that graph capture (~0.88 GiB) and the sampler warmup genuinely consume. Disabling the accounting just lets KV greedily take the space, then warmup has nowhere to go. The advertised 67,444-token pool was never spendable.

**Recipe 2 — raise util to 0.9907 (the boot log's recommended number), estimate on.** **Rejected at init**, before model load:

```
Free memory on device cuda:0 (23.05/23.56 GiB) on startup is less than
desired GPU memory utilization (0.9907, 23.34 GiB).
```

The cards were verified idle (1 MiB used, no zombie process). The gap is the ~0.5 GiB CUDA context each rank establishes before the util check runs — it eats into the free pool, so 0.9907 asks for more than is actually free after the process exists. The boot log's recommendation is computed from the tax arithmetic without accounting for that context overhead. On a 23.56 GiB 3090 it's simply above the headroom ceiling.

**What worked — laddering to the highest sustainable util.** Both canned recipes assume headroom this box doesn't have, so we found the real recovery point by hand. Util 0.97 boots clean: clears the init free-memory check, graph capture completes, warmup survives.

```
fixed MML 54496:
  util 0.95 (baseline)   avail KV 4.04 GiB   pool 54,496 tok   1.00x   serves
  util 0.97 (recovered)  avail KV 4.51 GiB   pool 60,854 tok   1.12x   serves
  env-off  @ 0.95        avail KV 5.0  GiB   pool 67,444 tok   1.24x   NON-VIABLE (warmup OOM)
```

So the **usable** recovered KV is +0.47 GiB ≈ +6,400 tokens *at fixed MML* — about half what the env-off boot advertised before it crashed. Advertised ≠ spendable; the functional-probe discipline is exactly what separates the two.

Pushing MML up at util 0.97, the engine solved the recovered ceiling the same way:

```
MML 63488:  pool 65,345 tok   1.03x
MML 65536:  pool 66,271 tok   1.01x   ← crosses the 64K tier (impossible at 0.95)
MML 67584:  FAILS — engine estimate of the ceiling: 66,848 tokens
```

Booted clean at 66,848 (max-conc 1.00×, graph capture complete) and **functional-probed** at ~90% (60K prompt):

```
prompt 59,860 tok → decode 32.3 / 32.3 tok/s, prefill ~1,030 tok/s, completes clean
```

**Recovered ceiling confirmed usable: 66,848 tokens, +12,352 over baseline (+22.7%).** Cost-model check at the recovered available-KV (4.51 GiB): predicted ~66,400, measured 66,848 → **+0.7%**. The two-coefficient KV model held across a +0.5 GiB shift in available memory.

The boot log at 0.97 noted that fully recovering the remaining tax would need util 1.0000 — i.e. the rest can't be reclaimed on this hardware without asking for all of VRAM, which the init check would reject. So 0.97 is near the practical recovery ceiling for this box.

## PP=4: an architecture-bound ceiling, and why the pool grows with MML

PP=4 (steered device-order 0,2,1,3, util 0.95) behaves completely differently. The pool is so large it outruns MML the whole way up, so KV never becomes the binding constraint:

```
MML  98304:  pool 268,617 tok   2.73x
MML 163840:  pool 311,332 tok   1.90x
MML 245760:  pool 338,240 tok   1.38x
MML 262144:  pool 341,935 tok   1.30x   ← 256K accepted (not clamped), max model len = 262144
```

Placement confirmed: all four GPUs resident; GPU3 carries ~22.1 GiB vs ~20.8 GiB on the others, consistent with it bearing the un-sharded LM head (embedding/LM-head don't shard under PP — the end stages are heavier).

**Why the pool increases with MML.** The reported number is a *capacity* (how many tokens fit), not a usage, and the available KV memory is roughly fixed across MML. Gemma 4 is a hybrid-attention model: sliding-window-attention (SWA) layers and global layers. The memory to hold one sequence of length L is

```
mem_per_seq(L) = global_per_token · L  +  swa_per_token · min(L, W)
```

where W is the SWA window. Once L exceeds W, the SWA term stops growing — it's pinned at the window. So the *amortized* per-token cost for L > W is

```
per_token_cost(L) = global_per_token + swa_per_token · (W / L)
```

As MML rises, the `W/L` term shrinks, the average per-token cost falls toward the global-only floor, and a fixed memory budget therefore holds more tokens. It is not that global layers eat into the pool slowly — it's that the fixed SWA window cost gets amortized over a longer sequence. The marginal token past W pays only the global-layer cost.

The four rungs match this amortization curve closely — close enough that it's the mechanism, not a coincidence. The curve also implies an asymptotic pool (the pure global-layer floor, as sequence length grows without bound) of ~409K tokens, which is academic here since the architecture caps at 256K.

(Caveat for Week 12: this sits on top of the known v0.21.0 HMA quirk — global layers are ~2× over-allocated because K=V unification isn't applied. If a later vLLM applies it, the global floor drops and the whole curve shifts up.)

**Functional probe at 235K (≈90% of 256K):**

```
prompt 234,454 tok → decode 15.1 / 15.1 tok/s, prefill 750.9 tok/s, completes clean
```

256K is usable in the sense that a quarter-million-token request completes. But decode is ~2.2× slower than TP=2's recovered ceiling, and the implied time-to-first-token is ~5 minutes (234K tokens / ~750 tok/s prefill, plus pipeline bubbles).

## The deployment read (corrected)

The first cut of this said "PP=4 covers the primary use case's context need where TP=2 cannot — it's the only config that serves the long-context end." That's wrong, or at least it scores the wrong thing. *Fit* is not the bar; *usable for the task* is.

The primary use case is an interactive root-cause-analysis loop: an operator investigating an incident. A multi-minute time-to-first-token fails that task no matter how much context loads. A 256K window an operator won't wait for is a spec-sheet number, not a deployment capability.

So the honest comparison is not "pick the winner on a context axis." The two configs bound different things:

```
config            ceiling      bound by        decode     TTFT @ ceiling   interactive?
TP=2 util 0.95     54,496      KV exhaustion    ~33 tok/s   seconds          yes
TP=2 util 0.97     66,848      KV exhaustion    ~32 tok/s   seconds          yes
PP=4 util 0.95    262,144      architecture     ~15 tok/s   ~5 minutes       no
```

**Neither single config serves the use case well.** TP=2 is interactive but context-limited (~67K recovered). PP=4 has the context but is not interactive. This is a "neither suffices alone" result, and that is precisely what motivates an orchestrator/sub-agent architecture rather than trying to make one config do both jobs:

- **Orchestrator: 31B TP=2 on the NVLink pair (GPUs 0,2)**, run at its interactive ceiling (~67K with the tax recovered). Fast decode, seconds-scale TTFT. Holds investigation state and reasons over distilled findings rather than ingesting everything itself.
- **Sub-agents: a smaller model (candidate: the new Gemma 4 12B-QAT) on single GPUs (the x1 cards, 1 and 3)**, fanning out the bulk-context tasks — reading long log spans and metric dumps in parallel — and returning only summaries to the orchestrator.

PP=4's 256K is not the deployment target. It is the *evidence* that brute-forcing context onto the 31B does not yield an interactive system — which is the argument for delegation. The long-context measurement earns its place by ruling out the simpler single-config architecture.

## What changed held-constant, and what didn't

- The tax-recovered config (util 0.97) is **not retroactive.** Days 1–5 and the PP=4 measurement all stay on util 0.95. Whether 0.97 becomes a Week 12 baseline is a deliberate decision for the end-of-week discussion, not something adopted by having measured it.
- PP=4 was measured at baseline util 0.95 so the TP-vs-PP comparison is apples-to-apples. The optional recovered-tax PP=4 run was not done (not needed for today's finding; PP=4 is architecture-bound, so more KV wouldn't move its ceiling).

## Open items carried to the end-of-week / Week 12 discussion

1. **Adopt the tax-recovered util (0.97) as a baseline?** It buys +22.7% usable context on TP=2. It changes held-constant, so it's a deliberate choice, and it would require re-baselining. The two boot-log-recommended recovery recipes are non-viable on this hardware — document that 0.97 (found by laddering) is the working path.
2. **12B sub-agent viability is now the load-bearing open question** for the orchestrator/sub-agent architecture. `google/gemma-4-12B-it-qat-w4a16-ct` failed to load on vLLM 0.21.0 (unsupported `gemma4_unified` architecture). This is the first Week 12 task: the single-GPU load test, gated on finding a vLLM release that supports both `gemma4_unified` and the 31B Dense path.
3. **vLLM version investigation** (0.22.1 / K=V-unification status), with a 31B re-baseline. Today's measurements are the regression baseline.

## Deliverables produced today

- `tools/start-vllm.sh` — added `--profiler-cudagraphs {on,off}` (committed before any probe).
- TP=2 ceilings: baseline 54,496 (util 0.95) and recovered 66,848 (util 0.97), both functional-probe-confirmed. Result JSONs committed.
- PP=4 ceiling: 262,144 (256K architectural, util 0.95), functional-probe-confirmed at 235K. Result JSON committed.
- CUDA-graph tax: quantified, and both boot-log recovery recipes shown non-viable on this box; working recovery path identified (util 0.97).
- Cost-model errors logged: TP=2 baseline −1.5%, TP=2 recovered +0.7%.
- Amortization model for the PP=4 pool-vs-MML behavior, confirmed against all four rungs.
