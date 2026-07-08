# Week 15 — Day 2: cross-tier interference — execution, verdict, and a mechanism correction

**Date:** 2026-07-08
**Session type:** execution (Claude Code). Command-heavy, mechanistic. No design decisions —
those were locked in `week-15-day1-interference-design-predictions-journal.md` (committed).
**Substrate:** frozen production stack — 4× RTX 3090, pinned `vllm/vllm-openai:v0.23.0`
(`sha256:6d8429e3…22ed8f`). 31B-QAT orchestrator TP=2 on GPUs 0+2 (NVLink pair, :8000);
two 12B-QAT workers TP=1 on GPUs 1 and 3 (:8001, :8003); nginx front door (:8080);
irs-prometheus + node-exporter.
**Provenance:** all results carry tool SHA `T@88493e1`. Result commits (R): boot `eee63a8`,
baselines `079fecf`, R2 `825b3c8`, R3 `8617dbe`.

---

## Headline

**Both tiers are isolated.** Under sustained saturation of the other two tiers, neither the
31B orchestrator nor a 12B worker loses more than **~0.5% of decode throughput** at any prompt
size. The two-tier box works as a **concurrent system**, not three configurations that each
boot. This is the operational proof the delegation architecture has owed itself since Week 11.

The result is the *more informative* of the two shapes Day 1 laid out: not flat-zero everywhere,
but **"isolated, with a host-handoff-rate signature that decays toward the context ceiling"** —
visible on the x1 worker (R3), concentrated at short context, decaying to ~0 near-ceiling.

One honest correction to the committed mechanism (below): the signature is **CPU-scheduling /
system-time contention**, *not* the softirq / interrupt-rate channel Day 1 co-hypothesized.
Softirq stayed at ~0.00% throughout. The "rate" character of the prediction held; the specific
host resource named was half wrong.

---

## For a reader new to the terms

**Prefill** = the one-time cost of reading the prompt (tokens/s). **Decode** = the steady
per-token generation rate after the reply starts (tokens/s). **c=1** = single-stream (one
request at a time) — the latency-sensitive operating point. **Victim** = the tier we measure
for slowdown. **Aggressor** = the tier(s) we saturate to try to cause that slowdown. Because
the four GPUs are physically separate and can't contend on compute or GPU memory, any
interference must flow through **shared host resources** — CPU cores, the PCIe root complex,
system RAM. That is the whole question of this week: is the host a hidden coupling channel?

---

## What ran (and in what order)

Strict sequence, one experiment per boot, predict-before-measure, commit-before-run:

1. **Tool changes first** (committed clean before any results run):
   - **worker2 port reconciliation → 8003.** `start-vllm.sh`'s preset and both READMEs said
     8002; `start-stack.sh` boots worker2 on 8003 and `interference_probe.py` floods :8003.
     A silent-404 hazard: a flood to a non-listening 8002 would *fake an isolation result*.
     Canonicalized on 8003 in `start-vllm.sh` (preset + two header examples), T's `README.md`,
     and R's `CLAUDE.md`. The `vllm-bringup-checks.sh` usage example (historical
     `gemma4-12b-bf16` container, not the worker2 preset) was left as-is. (`T@88493e1`, R
     `0a992de`.) Empirical port confirmation after boot — ground truth over preset — showed
     worker2 serving on :8003.
   - **Loaded-window timestamps in `interference_probe.py`.** Added a `loaded_window` block
     (flood-up, victim-probe start/end as ISO + unix epoch, flood-stop) so per-core CPU can be
     pulled from Prometheus over exactly the loaded interval. No change to load or measurement.

2. **Boot** — `start-stack.sh staggered --week week-15 --image …:v0.23.0` **with a required
   override** (see the stale-default finding below): `--model-31b google/gemma-4-31B-it-qat-w4a16-ct
   --mml-31b 131072 --util-31b 0.95`. All three tiers healthy (t2h 91/83/103 s).

3. **Empirical placement** (`vllm-bringup-checks.sh`, UUID→PID→physical-GPU): worker1→GPU1,
   worker2→GPU3, orchestrator→GPUs 0+2 (two PIDs, one per NVLinked card). All `placement_ok`.
   Boot-log KV ceilings: 12B **370,575 tok**, 31B-QAT **193,837 tok** (1.48× at 131K/req) —
   the 49,152 near-ceiling probe point is servable at c=1 on all three with wide headroom.
   (Day-1's "~54K ceiling" figure for the 31B was inaccurate; the empirical ceiling is 193,837.
   Ground truth over preset; 49,152 stands either way.)

4. **R1 solo baselines** — `throughput_sweep.py` c=1, max-tokens 256, iterations 5, warmup 1,
   sizes [512, 4096, 49152], **two runs per victim** for the run-to-run spread gate.

5. **R2** (31B victim, workers flooded via the nginx pool) and **R3** (12B victim, other worker
   + 31B flooded directly), each with a concurrent GPU-util sampler (sustained-saturation gate)
   and a post-run per-core CPU pull over the recorded loaded window.

---

## Two findings that would have voided the run if missed

**1. `start-stack.sh` ships a stale orchestrator default.** Its `MODEL_31B` default is the
old Week-11/13 `RedHatAI/gemma-4-31B-it-FP8-block` at `MML_31B=33024`. The frozen production
orchestrator is **31B-QAT** at MML 131072 (per `CLAUDE.md`, `start-vllm.sh`'s own orchestrator
preset, and the Day-1 design). Booting the default would have (a) measured the wrong model and
(b) capped MML at 33024 — **unable to serve the 49,152 near-ceiling point**, silently breaking
the entire near-ceiling regime *and* the R3 falsifiable commit. Worked around this run by
overriding via flags (same pattern the pickup uses for the stale `--week` default). **Recommend
a tool fix**: update `start-stack.sh`'s `MODEL_31B`/`MML_31B`/`UTIL_31B` defaults to the QAT
production config, or make the FP8 config an explicit opt-in — this default is a more dangerous
version of the port trap the session opened by fixing.

**2. The nginx pool was already reconciled to 8003.** The live `workers` least_conn upstream
targets `127.0.0.1:8001` and `127.0.0.1:8003` (config test passes) — verified before R2 so the
pool flood actually reaches both workers rather than half-502'ing on a stale 8002.

---

## R1 — solo baselines and the threshold escalation

Baselines (canonical run, tok/s):

| victim | metric | 512 | 4096 | 49152 |
|---|---|---|---|---|
| 31B-QAT | decode | 66.02 | 61.77 | 44.46 |
| 31B-QAT | prefill | 1958 | 1862 | 1140 |
| 12B-QAT | decode | 79.43 | 74.68 | 55.05 |
| 12B-QAT | prefill | 2545 | 2625 | 1588 |

**Run-to-run spread (the escalation gate):**

- **Decode is clean** on both victims — max spread 0.96% (31B) / 0.38% (12B), all cells ≤1%.
- **Prefill is noisy at short context** — prefill@512 spread **3.96%** (31B) / **4.04%** (12B);
  31B prefill@4096 1.74%; all other prefill cells clean. This is the c=1 short-prompt prefill
  jitter Day 1 anticipated (TTFT ≈ 0.25 s, so small scheduling jitter swings tok/s hard).

Per the Day-1 **committed** escalation ("threshold = 2× solo-baseline noise half-width; escalate
if any baseline shows >1.5% run-to-run spread"), the prefill isolation bar becomes
**max(3%, spread)**: ≈**4% @512**, **3% @4096/49152**. Decode keeps the clean **≤3%** bar.
**Decode is therefore the primary isolation signal; short-context prefill wobble is not read as
interference.** This is applying a pre-authorized contingency, not a new decision.

---

## R2 — 31B victim, both workers saturated → ISOLATED

Aggressor flood through the nginx pool (:8080); 1342 requests, 0 failures. **Saturation gate
PASS, sustained** (not just the probe's point-sample): GPU1 and GPU3 both min 97% / mean 98.7%
across the 357 s loaded window (concurrent nvidia-smi sampler).

| metric | 512 | 4096 | 49152 | bar | verdict |
|---|---|---|---|---|---|
| decode degradation | 0.05% | 0.11% | 0.25% | 3% | **isolated** |
| prefill degradation | 2.31% | 0.43% | 0.53% | 4% / 3% / 3% | **isolated** |

decode ≤0.25% everywhere; prefill@512 2.31% sits inside the ~4% baseline-noise band. The 31B is
the heaviest CPU citizen on the box (TP=2 + NCCL) and **dominates** the contended path rather
than being its victim — exactly as argued.

**R4-in-R2 (nginx least_conn split): 680 / 663** (8003/8001) = 50.6 / 49.4 — essentially even,
now confirmed on the v0.23.0 boot path.

---

## R3 — 12B victim, other worker + 31B saturated → ISOLATED; falsifiable commit HELD

Victim worker1 (:8001); worker2 (:8003) and the 31B (:8000) flooded directly (32 in-flight, no
pool); 640+529 requests, 0 failures. **Saturation gate PASS, sustained**: GPU0/GPU2 100%, GPU3
min 97% / mean 98.9% over the 264 s window.

| metric | 512 | 4096 | 49152 | bar | verdict |
|---|---|---|---|---|---|
| decode degradation | 0.41% | 0.49% | 0.06% | 3% | **isolated** |
| prefill degradation | −3.94% | −0.15% | +0.27% | 4% / 3% / 3% | **isolated** |

(prefill@512 −3.94% = loaded ran nominally *faster* than the canonical baseline — pure baseline
scatter within the 4% band, not a speedup.)

**Falsifiable commit — "near-ceiling degrades *less* than 512" — HELD** on the clean decode
signal: **0.06% @49152 vs 0.41% @512**. Near-ceiling is decisively the least exposed, so the
host-handoff-**rate** mechanism is *not* falsified: near-ceiling does the fewest host round-trips
per unit time (each decode step is longer, more GPU-bound on long-KV attention, so any fixed
per-step host delay is a smaller fraction). R3 is also the **largest** cell in the matrix
(decode ~0.4–0.5% vs R2's ≤0.25%), as predicted.

---

## Host-side corroboration (the mechanism)

Per-core CPU from node-exporter/Prometheus over each loaded window, vs a pre-flood idle window:

| window | mean core busy | busiest cores | softirq | MemAvailable |
|---|---|---|---|---|
| idle reference | 1.0% | — | ~0.00% | 40.8 GiB |
| R2 loaded | 41.1% | cores 6–9 ~60% | ~0.00% | 40.8 GiB (swing 0.04) |
| R3 loaded | 46.2% | cores 7–8 ~76% | ~0.00% | 40.7 GiB (swing 0.06) |

Reads cleanly: host CPU rises 1% → 41–46% under load (system-time to 5–15% on the busy cores),
**RAM dead flat**, **no core pegged** (max ~77%). Flat RAM rules out memory-bandwidth/pressure;
unpegged cores with headroom explain why the victims stay isolated — the shared host resource is
*loaded but not exhausted*, so per-token victim scheduling is never starved. R3's higher host
load (46% vs 41%, +31B aggressor) tracks its marginally larger victim degradation — the coupling
is real and monotone, just far below the 3% material line.

**Mechanism correction.** Day 1 named two channels: CPU-core contention **and**
softirq/interrupt-rate (many small DMA completions from busy aggressor GPUs). Observed: the
CPU-core/system-time channel is live; **the softirq channel is not** (softirq ~0.00% idle and
loaded). The interference that exists is compute-scheduling handoff — tokenize/detokenize,
sampling dispatch, SSE streaming, NCCL coordination sharing 6 cores — not interrupt handling.
The *rate*-dependence prediction (signature decays with prompt size) still held; the specific
resource was half wrong. Recorded as a resolution, not an edit to the committed journal.

---

## Prediction scorecard

| # | Committed prediction | Observed | Result |
|---|---|---|---|
| R2 | <1% decode/prefill @512·4096; ≤3% near-ceiling; isolated | decode ≤0.25%; prefill@512 2.31% (in noise), @4096 0.43%, @49152 0.53% | **held** (isolated); prefill@512 nominal >1% but within baseline noise |
| R3 | ~0.5–1% decode @512, decaying toward ceiling; largest in matrix; isolated | decode 0.41→0.49→0.06%; largest in matrix; near-ceiling lowest | **held**; @512 (0.41%) slightly *below* the 0.5–1% band |
| R3 falsifiable | near-ceiling degrades **less** than 512 (else mechanism wrong) | 0.06% @49152 < 0.41% @512 | **held** — mechanism not falsified |
| R4-in-R2 | nginx split ≈ 50/50 | 680/663 (50.6/49.4) | **held** |
| Matrix | isolated overall; signature concentrates short-context, decays with size | both isolated; R3 short-context signature decaying to ~0 near-ceiling | **held** |

**Recorded prediction misses / refinements (resolutions, not edits):**

1. **Mechanism (the real miss):** softirq/interrupt-rate channel co-hypothesized but **not
   observed** (softirq ~0). Channel is CPU-scheduling/system-time contention. Rate-dependence
   held; resource identity corrected.
2. R3 fine ordering: "concentrated at 512" implied 512 ≥ 4096; observed 4096 (0.49%) marginally
   ≥ 512 (0.41%). Both within decode noise (<1%); the material trend (decay to ~0 near-ceiling)
   held. Not a mechanism problem.
3. R2 prefill@512 predicted <1%, nominal 2.31%. Within the escalated ~4% noise band → consistent
   with isolation; the <1% point-prediction was tighter than the instrument's prefill noise floor.
4. R3 decode@512 came in 0.41%, just under the predicted 0.5–1% — slightly better isolation than
   predicted.

---

## Deliverables / state

- Results committed under `phase-3-.../week-15/results/` (T SHA `88493e1` in every artifact):
  boot choreography, four solo baselines (2 runs × 2 victims), R2 + sweep, R3 + sweep.
- Tool fixes committed in T (`88493e1`) and the CLAUDE.md doc-sync in R (`0a992de`) **before**
  any results run — clean SHA in every artifact.
- R / T / IRS clean. **vLLM stack torn down** (`start-stack.sh teardown`); observability stack
  (nginx + prometheus + node-exporter) left up — the documented idle condition restored, GPUs
  back to 0 MiB / 0%.

## Carry-forwards (not this session)

- **Tool fix:** `start-stack.sh` stale `MODEL_31B`/`MML_31B` default (finding above).
- **Delegation-architecture write-up finalization** (web chat) — the interference data now
  exists to close the operational-proof section and Week 11's "neither config serves the use
  case."
- **Week 16** — program conclusion (renames → consolidation → capstone → method Pulse), separate
  week / separate chat.
