# Week 15 — Day 1: cross-tier interference — design of record + committed predictions

**Session type:** design / prediction (web chat). No measurement taken this session.
**Substrate:** frozen production stack — 4× RTX 3090, pinned `vllm/vllm-openai:v0.23.0`
(`sha256:6d8429e3…22ed8f`). 31B-QAT orchestrator TP=2 on GPUs 0+2 (NVLink pair, :8000);
two 12B-QAT workers TP=1 on GPUs 1 and 3 (:8001, :8003); nginx front door (:8080);
irs-prometheus + node-exporter up.

This session locks the experiment shape and commits the predictions the execution session
(Claude Code) will score against. The measured runs are the next session.

---

## Scope preamble (carries with every prediction below)

This characterizes cross-tier interference on the **frozen 4× RTX 3090 topology**, where the
four GPUs are physically separate and cannot contend on compute or GPU memory — so the only
interference channel is **host-side**. It is a **terminal characterization of a retiring
topology**, not a migration baseline.

The single-die RTX PRO 6000 measures a *different mechanism* — intra-die SM / HBM-bandwidth /
L2 contention under continuous batching, with the nginx worker pool retired and the deployment
collapsed to a single endpoint. Its numbers **must not be trended against these**; a reader who
lines the two data sets up is committing a category error. What transfers to the successor
program is the *method* (predict-before-measure interference characterization) and the closed
operational-proof this gives the capstone — not the values. `[argued — mechanism trace: the
contended resource differs in kind (separate-die host-side vs. single-die on-chip), so the two
measurements share a name and nothing else.]`

The Week 13 Day 3 interference numbers are a **prior, not a prior result**: they were taken on
the *pre-v0.23.0-convergence* worker-boot path (the native 12B boot landed Day 4/5), at 512 and
4096 only. They inform the predictions; they are not being re-confirmed. `[settled]`

---

## Design of record (locked)

**Regimes.**

- **R1 — per-tier isolated baselines.** The reference each diff is taken against. Re-established
  on the frozen v0.23.0 config; Day 3's baselines (old boot path) are not reused. `[settled]`
- **R2 — 31B victim, workers saturated** (`--victim 31b`), aggressor load through the nginx pool.
  Covered by `interference_probe.py` as-built. The pool-flood also carries the nginx
  upstream-split capture, so **R4 is folded into R2**. `[settled]`
- **R3 — 12B victim, other worker + 31B saturated** (`--victim 12b`). Covered as-built. `[settled]`
- **R4 — full concurrent through nginx.** Folded into R2: the pool path already exercises the
  front door and records the upstream split; a dedicated symmetric-load harness would yield a
  system-throughput observation, not an interference diff. `[argued — rejected alternative: a
  standalone R4 harness. Recorded in case a genuine all-tiers-hot system snapshot is wanted later.]`

**Victim probe sizes:** 512, 4096, **and a near-ceiling point** (proposed 49,152; confirm each
victim serves it at c=1 from the boot-log KV ceiling before committing the baseline). The
near-ceiling point covers the actual use-case operating region, which Day 3 never probed. `[argued]`

**Isolation metric + threshold (set blind):** victim decode and prefill tok/s under load as a
ratio to the solo baseline, per prompt size; degradation = 1 − ratio. **Isolated iff decode
*and* prefill degradation ≤ 3% at every prompt size.** 3–10% = watch; >10% = material. The 3%
floor sits one point above warm-up-controlled decode noise (Day 3 caught a 4–7% *warm-up*
artifact on one prefill point; the re-measured-baseline discipline neutralizes it). `[argued —
rejected alternative: threshold = 2× measured solo-baseline noise half-width; escalate to it
if any solo baseline shows >1.5% run-to-run spread before trusting a "material" verdict.]`

**Saturation gate:** ≥ 95% sustained aggressor-GPU util per aggressor GPU across the post-ramp
sample window; any aggressor GPU below that **voids the run**. Day 3's flood defaults drove
aggressor GPUs to 98–100%, so this is a floor the defaults clear. `[settled]`

**Host-side corroboration:** scrape **per-core CPU / softirq time** (node-exporter → Prometheus)
across the loaded window. The mechanism is scheduling/handoff contention, so pegged cores
alongside flat RAM is the confirming signature; `free -m` alone is insufficient. `[argued]`

---

## Committed predictions

Mechanism (shared by R2/R3): with disjoint GPUs, the only channels are **host CPU-core
contention** (three vLLM servers — the 31B's TP=2 is two GPU-worker processes plus NCCL
coordination — sharing 6 cores of a 9600X with nginx/prometheus/node-exporter; per-token
scheduling, tokenize/detokenize, sampling dispatch, SSE streaming) and **root-complex
transaction/interrupt *rate*** (many small DMA completions from busy aggressor GPUs consuming
softirq handling). **Not PCIe bandwidth** — steady-state serving moves only token IDs
(a 50K-token prompt ≈ 200 KB, sub-ms even on Gen4 x1), so the x1 link's narrowness governs
*model-load* time, not serving. **Not NVLink** — the 31B's TP=2 all-reduce rides NVLink between
GPUs 0+2, off the PCIe fabric entirely.

Both live channels scale with **token/request rate**, so the single-stream (c=1) victim is most
exposed where it does the most host round-trips per unit time — **short-context decode** — and
least exposed near-ceiling, where each decode step is longer GPU-bound on long-KV attention and
any fixed per-step scheduling delay is a smaller fraction of a slower step.

| Regime | Victim | Aggressors | Predicted degradation | Verdict |
|---|---|---|---|---|
| R2 | 31B (GPUs 0+2) | both workers | <1% decode/prefill @512·4096; ≤3% @near-ceiling | isolated |
| R3 | 12B (x1 worker) | other worker + 31B | ~0.5–1% decode @512 (Day 3: 0.5–0.7%), **decaying toward ceiling** | isolated |
| R4-in-R2 | nginx split | — | ≈ 50/50 (Day 3: 190/190) | even |

**R2.** `[argued — high confidence @512/4096, medium @near-ceiling]` The 31B is the *heaviest*
CPU citizen on the box (TP=2 + NCCL), so it dominates the contended path rather than being its
victim. Least-confident cell: near-ceiling prefill (the one place the 31B streams large host→GPU
concurrently with the workers' load) — still predicted ≤3%.

**R3.** `[argued — the flagged watch prediction]` Largest degradation in the matrix, concentrated
at **512, not near-ceiling**, because this victim's aggressor set includes the 31B and short
context maximizes host-handoff rate. **Falsifiable commit: near-ceiling degrades *less* than
512.** If near-ceiling instead shows the largest degradation, the CPU/transaction-rate mechanism
is wrong and something prompt-size-dependent (unmodeled) is in play.

**R4 (folded).** `[argued]` least_conn + symmetric workers + uniform aggressor requests → even
split; nginx sits upstream of vLLM and is blind to the v0.23.0 boot-path change. Held exactly on
Day 3, but on the old worker path, so not `settled`.

**Matrix-level.** `[argued]` Near-ceiling does not flip the verdict (isolated overall); any
signature concentrates at short context and *decays* with prompt size. Two distinguishable
outcomes: flat ≤1% everywhere = strong isolation; a 512-concentrated signature on the x1 worker
that decays toward near-ceiling = "isolated, with a host-handoff-rate signature" — the more
informative result, whose declining-with-prompt-size shape is itself the mechanism confirmation.

---

## Next session (Claude Code — execution)

Frozen-config boot on pinned v0.23.0 → empirical placement (UUID→PID→cgroup) → solo baselines
established-and-committed on v0.23.0 at [512, 4096, 49152] → probe matrix R2/R3 → per-core CPU
capture over the loaded window → diff vs. committed baselines. Gate list is the session's pickup
prompt (not committed).
