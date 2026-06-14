# Week 13 — Day 3: Cross-tier interference under load

**Date:** 2026-06-14
**Phase:** 3 — Optimization & Quantization (Weeks 11–14)
**Host:** `inference` (4× RTX 3090; GPUs 0+2 NVLink-bridged, GPUs 1+3 on PCIe 3.0 x1)
**Repos touched:** `rtx3090-ai-training` (several commits — launcher fix, interference harness,
baselines, two interference directions, baseline recheck); `inference-reference-stack`
(no commits; an orphaned results directory was deleted)

---

## What this day was

Three measurements on the now-co-resident two-tier stack — a larger "orchestrator" model
(Gemma 4 31B, served across the two NVLinked GPUs) plus two smaller "worker" models
(Gemma 4 12B QAT, one per PCIe-x1 card):

1. **APC verification** — does *automatic prefix caching* (reusing the stored attention
   state of a repeated prompt prefix instead of recomputing it) actually work on the
   hybrid-attention Gemma 4 12B worker?
2. **Cross-tier interference** — if two of the three serving tiers are driven to full
   saturation, does the third tier's latency degrade, and by how much?
3. **least_conn split** (folded into #2) — when concurrent traffic is sent through the
   nginx front door's load-balanced pool, how does it distribute across the two workers?

All three came back matching the predictions made before measuring. There was also one
methodology catch (a warm-up artifact in the first baseline) and a fair amount of
pre-session git untangling.

For a reader new to the terms used throughout: **prefill** is the one-time cost of reading
the prompt (measured in tokens/second); **decode** is the steady per-token generation rate
after the model starts replying; **TTFT** is time-to-first-token; **TP=2** means the 31B's
weights are split ("tensor-parallel") across two GPUs that must exchange data every layer;
**KV cache** is the per-sequence memory holding attention state.

---

## Pre-session: housekeeping and a git knot

Several bookkeeping issues surfaced and were resolved before any measurement:

- **Orphaned results in the wrong repo.** A handful of boot-choreography log files had
  landed in `inference-reference-stack` instead of `rtx3090-ai-training`. Root cause:
  `tools/start-stack.sh` resolved its results path from `git rev-parse --show-toplevel`,
  which returns the top of *whichever* checkout the current directory sits in — so running
  the script from inside the IRS checkout retargeted its output (and its recorded git SHA,
  and even its launcher lookups) at the wrong repo. The canonical Day-2 results were already
  committed in the correct repo, so the IRS copies were confirmed orphans and deleted.

- **Launcher fix (committed before any measured run).** `start-stack.sh` was changed to
  anchor its repo root to the **script's own location** (`BASH_SOURCE`) rather than to the
  current directory. This is invariant both to where the script is run from and to where the
  repo is cloned — the latter matters because IRS is meant to be portable. One-line behavioral
  change; everything downstream already read the anchored variable.

- **Boot file clobbered by the morning re-boot.** Bringing the stack up via
  `start-stack.sh simultaneous` overwrote the committed Day-2 simultaneous-boot result,
  because that launcher builds its output filename from mode + model names with **no
  timestamp** — so every simultaneous boot writes the same file. The diff confirmed it was
  only today's incidental re-boot (timestamps, SHA, ~2 s of boot-time jitter); the working
  copy was discarded to keep Day-2's committed numbers. (As a side note, the re-boot
  reproduced Day-2's boot timings closely, which is a small bonus confirmation that boot
  choreography is stable.)

- **Branch divergence.** The interference harness had been committed from a second machine
  and pushed; the solo baselines were committed locally. The two built on the same parent and
  touched disjoint files, so `git pull --rebase` reconciled them into a clean linear history
  with no conflict.

---

## Experiment 1 — APC verification on the 12B worker

**Setup.** Send the *same* ~32K+ token prompt prefix twice to a single worker (port 8001
directly), with **no per-request nonce** — this probe *wants* the cache to hit, which is the
opposite of the throughput sweeps, where a random nonce is prepended specifically to defeat
the cache. The question: does the cache fire, and is it observable?

A config check first, so a null result would be interpretable: the worker's startup log shows
`enable_prefix_caching=True` and emits a live `Prefix cache hit rate` metric. So caching is
*enabled* — any negative result would mean "the model can't," not "it was switched off."

**Result — APC works, decisively.** With an identical 54,019-token prefix:

| request | wall time | meaning |
|---|---|---|
| 1 (cold) | 35.25 s | full prefill of 54K tokens |
| 2 (warm) | 0.13 s | prefill skipped — served from cached attention state |

A ~260× collapse. The only thing that returns a 54K-token prompt in 0.13 s is a skipped
prefill. The engine's own `Prefix cache hit rate` metric corroborated by climbing off 0.0%.
A follow-up request after an unrelated warm-up ping was *still* warm, so the cached prefix
persists across intervening traffic — eviction is not aggressive at this footprint.

**The caveat that matters for routing design.** The per-response usage field
`prompt_tokens_details.cached_tokens` was **null** on every request. So APC works but does
**not report through the standard API response**. These are two different facts, and an
earlier framing had conflated them ("null cached_tokens ⇒ no caching ⇒ routing must ignore
it"). The corrected read: cache-affinity routing — pinning a conversation's follow-up turns
to the same worker to harvest that prefill skip — is viable on capability grounds. What's not
viable is a router reading cache state *from the API response*; it would have to scrape the
engine's `Prefix cache hit rate` metric (which the observability stack can expose) or infer
from latency.

**Prediction check:** predicted APC would fire on this newer image (the earlier null was on
the older 31B image). Held.

---

## Experiment 2 — Cross-tier interference under load

**The premise being tested.** GPU memory and GPU compute are **disjoint per tier**: the 31B
lives on GPUs 0+2, the workers on GPUs 1+3, and nothing is shared between those GPU sets. So
a busy aggressor *GPU* cannot, by construction, slow a victim on different GPUs. The only
channels left for interference are **shared host resources** — host CPU (each server's
scheduler, tokenizer, sampling, and streaming loop), system-memory bandwidth, and PCIe
traffic to the shared root complex. The prediction from Day 2 was therefore: interference
should be **mild, and host-side only**.

**Load shape.** The aggressor tiers were driven with *concurrent* load (many simultaneous
requests), not a single sustained stream. This is deliberate: a single stream pegs only the
aggressor GPU — the one dimension that *can't* transfer to the victim — whereas concurrency
is what actually loads the shared host path (scheduler busy every step, batched tokenization,
more memory and PCIe traffic). The victim, by contrast, was probed with a clean single-stream
latency measurement. Aggressor prompts were nonce-prefixed to defeat the prefix cache (proven
live in Experiment 1), so each aggressor request does real work rather than hitting cache.

The harness (`tools/interference_probe.py`, new today) verifies victim and aggressor GPU
placement empirically before loading, samples aggressor-GPU utilisation after a ramp as proof
the load actually landed, runs the victim probe via the existing `throughput_sweep.py` at
concurrency 1 (so the loaded measurement is on the identical code path as the solo baseline),
then diffs against the committed solo baseline.

> **Baseline note.** The first solo baseline measured this session ran ~4–7% low on prefill
> at the first prompt size (512) only — a warm-up artifact discussed in the methodology
> section below. A re-measured baseline is used for all comparisons here.

**Saturation was confirmed in both directions** (this is the gate that makes a null result
meaningful rather than a sign the load never arrived):

- 31B-as-victim: aggressor worker GPUs 1+3 at 98% / 100%; 379 aggressor requests, 0 failures.
- 12B-as-victim: aggressor GPUs 0+2+3 all at 100%; 284 aggressor requests, 0 failures.

**Results (loaded vs. re-measured solo baseline):**

*31B orchestrator as victim — workers saturated:*

| prompt | decode solo | decode loaded | ratio | prefill solo | prefill loaded | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 512  | 44.3 | 44.3 | 1.00 | 1945 | 1940 | 1.00 |
| 4096 | 42.2 | 42.2 | 1.00 | 1792 | 1793 | 1.00 |

*12B worker as victim — the other worker AND the 31B saturated:*

| prompt | decode solo | decode loaded | ratio | prefill solo | prefill loaded | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 512  | 79.3 | 78.6 | 0.99 | 2667 | 2656 | 1.00 |
| 4096 | 74.6 | 74.1 | 0.99 | 2633 | 2623 | 1.00 |

**Verdict: interference is negligible (≤1% on decode), even at full saturation.** The 31B is
completely insulated. The 12B worker takes a barely-measurable ~0.5–0.7% decode hit — and
this is the *expected* asymmetry: the victim worker sits on a bandwidth-starved PCIe-x1 card,
and its aggressor set includes the 31B, whose tensor-parallel cross-GPU traffic and larger
per-step host work is a heavier host-CPU load than two workers alone. So the only measurable
effect appears exactly where the mechanism predicts it should, at exactly the small magnitude
predicted.

The practical conclusion: this orchestrator + worker co-residency layout is **safe to run
with all tiers hot**. Interactive latency on any tier does not depend on what the other tiers
are doing.

**least_conn split (folded in).** The 31B-as-victim direction drove its worker load through
the nginx pool, which let the access log record where each request was routed. Over 380
concurrent pool requests the split was **190 / 190 — exactly even** across the two worker
upstreams (ports 8001 and 8003). Day 2 had shown that *serial* traffic pins to the first
backend; this confirms that under *sustained concurrency* with equal-capacity backends,
`least_conn` balances perfectly with no skew. Combined with Experiment 1, the routing picture
is now complete: `least_conn` handles load distribution cleanly on its own, and cache-affinity
would be a latency optimisation layered on top, not a balancing necessity.

---

## Methodology note: a prefill warm-up artifact at the first measured size

The interference test surfaced an apparent oddity — prefill at 512 tokens looked ~4–7%
*faster under load* than at the solo baseline, which is physically impossible for a contention
test (load cannot speed the victim up). Re-measuring the solo baselines resolved it: the
**original baseline's prefill at the first measured prompt size was low**, not the loaded run
being high. The recheck reproduced the higher figure on both models independently:

| model | metric | original baseline | recheck baseline |
|---|---|---:|---:|
| 31B | prefill @512 | ~1806 | ~1945 |
| 12B | prefill @512 | ~2559 | ~2667 |

Decode was identical across original, loaded, and recheck runs everywhere, and prefill at
4096 (a later size in the sweep) was stable throughout. So the artifact is narrow and
specific: **the first measured prompt size in a fresh sweep underreports prefill by a few
percent**, almost certainly because the GPU boost clock has not fully settled by then. The
existing `--warmup 1` clears the cold-start spike for *decode* (a known, previously-documented
effect) but does not fully settle *prefill clock-state* by the first measured size. By the
second size it has.

This did not affect the interference verdict — decode is the sensitive interactive metric and
it was clean, and prefill at 4096 was flat — but it is worth carrying forward as a tool
consideration (see Day-4 list).

---

## Prediction-error attribution: the "31B is faster than expected" surprise

The 31B solo baseline initially looked much faster than the remembered Week-11 figures
(~33 tok/s decode / ~1,130 tok/s prefill). It is not faster. Those remembered numbers are the
**context-ceiling operating point** — measured near the ~54,496-token maximum, where decode
and prefill have both degraded down the curve. Today's measurement was at 512 and 4096 tokens,
the fast end of the *same* curve. At matched prompt sizes, today ties Week-11 Day-2 almost
exactly (decode within 0.1 tok/s).

The error was entirely a reference-point mismatch: a single decode/prefill number for a model
is meaningless without stating the context length it was measured at. This is the existing
"rates are not constant across context — compare only at matched context" lesson recurring,
and worth restating because it is easy to fall into.

---

## Tooling

- **`tools/interference_probe.py` (new).** Cross-tier interference harness. One direction per
  invocation (`--victim 31b` / `--victim 12b`). Reuses `throughput_sweep.py` at c=1 for the
  victim probe; uses a built-in concurrent chat-endpoint flooder for the aggressors (necessary
  because the sweep posts to `/v1/completions`, while the nginx pool routes
  `/v1/chat/completions`). Verifies GPU placement before loading, samples aggressor-GPU
  utilisation as a saturation gate, captures the nginx upstream distribution, and diffs the
  loaded victim against its solo baseline. Model identities are discovered from each live
  endpoint and propagated into payloads, metadata, and default filenames — nothing
  model-specific is hardcoded. Diff and nginx-log parsing are best-effort so a parse hiccup
  cannot lose the core victim measurement.

- **`tools/start-stack.sh` (fixed).** Results/SHA/launcher paths now anchor to the script's
  own location instead of the current directory's git top-level (see Pre-session).

---

## Day-4 carry-forward

**`start-stack.sh` fix list (the cwd-anchor was the first of three):**

1. ~~cwd-anchor bug~~ — **done today.**
2. **Hardcoded `week-13` in the default results path.** Next week the launcher would write to
   the stale week's directory unless overridden. Make the week a parameter.
3. **No-timestamp boot filename clobbers committed results.** Each boot of a given mode writes
   the same filename, silently overwriting a prior committed result (this bit us this morning).
   Add a timestamp to the default filename, or a refuse-to-overwrite guard.

**Other touches flagged earlier and still open:**

- nginx directory-mount robustness (mount the config's directory, not the single file, so
  `nginx -s reload` survives git's inode swap).
- `start-stack.sh` dirty-tree warning false-positive on the results directory.
- `start-12b-qat.sh` `--gpu`-derived default port (removes a port-collision foot-gun).
- CUDA-graph-tax flag (`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`, flag-gated opt-in).
- **New from today:** consider whether the prefill warm-up artifact warrants a tool change
  (e.g. a throwaway probe at the *first* size specifically, or a dedicated prefill-warmup
  lever) versus simply being a documented caveat.

**Version-convergence work (v0.23.0 / native `gemma4_unified`) and the 31B re-baseline remain
the Day-4 headline**, with the FP8→QAT migration experiment gated behind a clean Day-4
re-baseline as Day 5.

---

## Status: Day 3 complete

- [x] APC verification — works (~260× on a repeated 54K prefix); `cached_tokens` unreported
      (a reporting gap, not a capability gap); routing implication recorded.
- [x] Cross-tier interference — ≤1% decode in both directions at full saturation; prediction
      held in direction and magnitude; co-residency layout validated as safe all-hot.
- [x] least_conn split — 190 / 190, even under sustained concurrency.
- [x] Prefill warm-up artifact identified, attributed, and worked around (re-measured baseline).
- [x] All result files committed; both repos clean; history linear and pushed.
