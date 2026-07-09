# The Delegation Architecture: A Capable Orchestrator with Cheap Workers

**Repo:** R (`docs/delegation-architecture.md`)
**Date:** 2026-07-08 (Week 15, Session 3)
**Scope:** the two-tier serving architecture developed and validated across Weeks 11–15, with
the Week 15 operational proof folded in. This document is the record of the architecture arc;
per-measurement detail lives in the cited daily journals.

---

## Thesis

A single model asked to be both the reasoner and the bulk-context reader will fail one of the
two jobs. The architecture that resolves this is **delegation**: a capable orchestrator that
holds investigation state and reasons over *distilled* findings, fanning bulk-context work out
to cheap, fast specialist workers that read the raw material and return summaries.

The thesis rests on several independent reasons, not one:

- **Context management.** The orchestrator never ingests the raw bulk. Workers absorb long log
  spans and return distilled findings, so the orchestrator's context window holds investigation
  state and conclusions rather than source material. Each tier's context budget is sized to its
  actual job.
- **Token cost / performance.** The bulk of the tokens — long prompts over raw logs — are
  processed on the cheap, fast tier. The expensive, capable tier spends its tokens only on
  reasoning over summaries. The cost and latency profile of the system tracks where the tokens
  actually flow, not the headline size of the largest model.
- **Interactivity.** Delegation keeps the operator-facing loop on a configuration that responds
  in seconds. This was the empirically forcing reason at design time: the Week 11 measurements
  below showed that no single configuration of the then-current orchestrator variant was
  simultaneously interactive and long-context on this hardware. The later QAT migration
  lifted this constraint without removing the other motivations — see the re-examination in
  the next section.

The thesis is **substrate-neutral**: capable-orchestrator + cheap-specialists is decoupled from
the deployment substrate. The same design runs on frontier APIs (a frontier model orchestrating
smaller hosted models) or on self-hosted weights, chosen per cost and privacy constraints. The
system documented here — Gemma 4 31B orchestrating two Gemma 4 12B workers on four consumer
GPUs — is the **proof case, not the thesis**.

**Use case.** The driving application is an **operator copilot** for root-cause analysis of a
distributed system: an interactive investigation loop in which the orchestrator plans, delegates
log-reading to workers scoped per component, and reasons over their findings.

---

## The case for delegation

Three constraints motivate delegation on this hardware class. Their standing differs — the
first and third hold on the production stack today; the second is the one that forced the
decision at design time and was later lifted:

1. **Context management and token cost.** An orchestrator that ingests raw logs spends its
   window on source material instead of investigation state, and routes the bulk tokens through
   the expensive tier. Both hold regardless of how large the window is.
2. **Context ceiling / interactivity.** On the orchestrator model quantization the program ran
   at design time (FP8), no single configuration was both interactive and long-context. This
   was the forcing constraint when the decision was made; the Week 13 QAT migration lifted it
   (below).
3. **Concurrency at depth.** At long context, responsiveness degrades toward serial: the
   orchestrator's max concurrency is 1.48× at the production operating point (MML 131,072), so
   a fan-out investigation reading several components' logs serializes on one model.

Unlike the first two, the third constraint was not a design-time premise — it was discovered in
testing. The Week 11 under-load characterization showed that the orchestrator blocks on the
slowest request in a fan-out wave, and the Week 12 worker characterization found the same shape
at the smaller tier: the 12B is functionally serial at 64K+ context (batching gains 2.33× at 8K
and effectively nothing at depth). At full context the orchestrator's KV pool holds 1.48×
concurrency — roughly one investigation's worth. A worker tier sidesteps this: several workers
read in parallel, each decoding faster per stream than the orchestrator (~74.6 vs ~42.3 tok/s
at matched 4K context).

The lifecycle asymmetry compounds the point: the orchestrator runs for the duration of an
investigation, accumulating state throughout, while workers are ephemeral — a worker receives a
task or a series of tasks, returns its findings to the orchestrator, and is discarded; a fresh
worker context is spawned for the next set of investigative tasks. A long-lived, near-serial
orchestrator is exactly the wrong place to queue bulk work; disposable parallel workers are the
right one.

Constraints 1 and 3 never depended on the second. Had the program started on the 31B QAT, the
ceiling would not have been binding — but we believe the same decision would have followed from
the other two. That is a counterfactual judgment, not a derivation from the record.

### The design-time measurement (Week 11, FP8)

At design time, the FP8 checkpoint was the 31B variant in play — the QAT w4a16 checkpoint
entered the stack two weeks later. Week 11 characterized the 31B Dense (FP8, vLLM 0.21.0) under
every viable parallelism layout on the 4× RTX 3090 box, looking for a single configuration that
serves the interactive RCA loop.

(**TP=N** — tensor parallelism, degree N: the model is split across N GPUs that share the work
on every layer. **PP=N** — pipeline parallelism, degree N: the model is split into N sequential
stages, one GPU per stage.)

| config | context ceiling | bound by | decode | TTFT @ ceiling | interactive? |
|---|---:|---|---:|---|---|
| TP=2, util 0.97 | 66,848 | KV exhaustion | ~32 tok/s | seconds | yes |
| PP=4, util 0.95 | 262,144 | architecture | ~15 tok/s | ~5 minutes | no |

TP=2 is interactive but context-limited (~67K). PP=4 has the
context but a ~5-minute time-to-first-token at depth — it fails the interactive task no matter
how much context loads. No configuration delivered what the RCA loop needs: a sufficiently
large context window *and* acceptable decode rate *and* acceptable time-to-first-token.

PP=4's 262K ceiling is therefore not a deployment target. It is the **evidence** that
brute-forcing context onto the large model does not produce an interactive system — which, at
design time, closed the case for delegation. The rejected alternative was single-config
scale-up; the long-context measurement earns its place by ruling out that simpler design.

(TP=2 additionally beat PP=4 on aggregate generation throughput and fan-out completion time at
every prompt size under load — the delegation-relevant metric, since the orchestrator blocks on
the slowest worker in a wave.)

### The QAT migration lifted the second constraint

The Week 13 move to the QAT orchestrator roughly tripled the TP=2 KV budget: the empirical
ceiling at the production operating point (MML 131,072, util 0.95) is **193,837 tokens** — the
Week 13 ceiling walk, reproduced to the token in Week 14 — and the same walk showed MML 262,144
refused at that util (estimated max ~218,624). A single QAT orchestrator serves the validated
131K envelope interactively, so the ceiling no longer forces the design. Constraints 1 and 3
are unaffected — on the production QAT stack they carry the motivation, and Week 11's
measurement stands as the historical record of what forced the decision when it was made.
(This re-examination is this document's own synthesis of the Week 13/14 numbers against the
Week 11 argument.)

---

## The proof-case system

The self-hosted deployment, frozen for this program's conclusion:

| tier | model | layout | GPUs | port |
|---|---|---|---|---|
| orchestrator | `google/gemma-4-31B-it-qat-w4a16-ct` | TP=2 | 0 + 2 (NVLink pair) | 8000 |
| worker 1 | `google/gemma-4-12B-it-qat-w4a16-ct` | TP=1 | 1 (x1 riser) | 8001 |
| worker 2 | `google/gemma-4-12B-it-qat-w4a16-ct` | TP=1 | 3 (x1 riser) | 8003 |
| front door | nginx (`least_conn` pool + named routes) | — | — | 8080 |

Engine: pinned `vllm/vllm-openai:v0.23.0` (manifest digest `sha256:6d8429e3…22ed8f`), one image
for both tiers, zero per-model workarounds. Deployed MML 131,072 at util 0.95 (orchestrator) and
0.90 (workers) — the asymmetry is need-driven: at 0.90 a worker's KV pool already holds the
12B's full 262,144-token architectural context at 2.16× concurrency, so raising it buys
nothing, while the 31B refused to boot at 0.90 during the Week 13 bring-up (the CUDA-graph
memory reservation left too little KV to admit even the smaller Week 11 baseline pool), making
0.95 the characterized working value. The orchestrator's empirical KV ceiling at that operating
point is 193,837 tokens (max concurrency 1.48×; reproduced live in Week 14). GPU placement is
verified empirically per boot (`nvidia-smi` UUID→PID→cgroup join), never assumed from launcher
intent.

The front door exposes a worker pool (`/v1/…` → least_conn across both workers) and named
instance routes (`/v1/worker/N/…`, `/v1/orchestrator/1/…`), so callers can say either "any free
worker" or "this one." The routing contract was verified in Week 13.

The deployment matches each tier to the interconnect it actually needs: the tensor-parallel
orchestrator sits on the NVLink pair (its all-reduce traffic rides NVLink, off the PCIe fabric
entirely), and the workers sit one-per-card on the x1 risers, where a single-GPU model is
unaffected by link width at serving time (steady-state serving moves token IDs, not tensors).

---

## Validation record

**Orchestrator tier (Weeks 11 and 13).** The 31B runs TP=2 on the NVLink pair, characterized
across parallelism layouts in Week 11 (previous section), then migrated FP8 → QAT w4a16 in
Week 13: the QAT checkpoint loads and serves on the pair, with the empirical KV ceiling walked
to 193,837 tokens at the production operating point, and decodes 36–50% faster than the FP8
variant across the prompt-size ladder (bandwidth-bound decode; prefill +1.8–3.9%).

**Worker tier viability (Week 12).** The 12B QAT checkpoint loads and serves on a single 24 GB
card (8.28 GiB weights). No memory ceiling exists on the card — the full 262,144 architectural
context fits at 2.16× concurrency. Production MML is pinned to 131,072, the model's validation
boundary at the time of the Week 12 validation; the 131K–262K range fits in memory but is
quality-unvalidated, an open item carried from Week 12. Measured single-card throughput: decode
69.6 tok/s @8K / 51.7 @64K / 46.2 @102K; batching gains 2.33× at 8K but the worker is
functionally serial at 64K+ — a direct input to the front-door design (at depth, queueing ≈
batching, with better latency).

**Quality (Week 13).** QAT w4a16 is use-case-equivalent to the BF16 parent at both orchestrator
and worker tiers, via position-bias-controlled LLM-as-judge evaluation: guardrail adherence an
8/8 tie at the orchestrator; pointwise 4.83–5.0 at the workers across both probe components;
format conformance 6/6. The quantization that makes the two-tier layout fit this hardware costs
no measured quality.

**Co-residency (Week 13).** All three services boot and serve concurrently on the one host:
~40 GB of 64 GB host RAM to spare, no swapping; simultaneous boot ~2.5× faster than staggered.

What Week 13 could *not* yet claim was that the tiers stay out of each other's way under load —
the three services share host CPU, the PCIe root complex, and system RAM even though they occupy
disjoint GPUs. That is the operational proof, and it is what Week 15 measured.

---

## The operational proof (Week 15)

Terminology: the **victim** is the tier measured for slowdown; the **aggressors** are the
tier(s) saturated with load to try to cause that slowdown. Because the four GPUs are physically
separate, any interference must flow through shared host resources — which is the question the
experiment asks.

Method: committed predictions per tier and regime before any measurement (Day 1 journal), solo
baselines re-measured on the frozen v0.23.0 stack, then two loaded regimes — R2 (31B victim,
both workers saturated) and R3 (12B victim, other worker + 31B saturated) — at prompt sizes 512,
4096, and 49,152, with a pre-committed isolation bar of ≤3% decode *and* prefill degradation at
every size, and a ≥95% aggressor-GPU-utilization gate on every run. Provenance: tool SHA
`T@88493e1`; result commits R `eee63a8` (boot), `079fecf` (baselines), `825b3c8` (R2),
`8617dbe` (R3).

**Verdict: neither tier slows the other down.** Under sustained saturation of the other tiers,
neither the orchestrator nor a worker loses more than ~0.5% of decode throughput at any measured
prompt size — a ~7× margin inside the pre-committed 3% isolation bar, so both tiers qualify as
isolated in the experiment's defined sense. Before this measurement, the record showed only
that the three services boot and run together; it now shows they sustain full concurrent load
without degrading one another.

| regime | victim | decode degradation (512 / 4096 / 49,152) |
|---|---|---|
| R2 | 31B orchestrator | ≤ 0.25% at all sizes |
| R3 | 12B worker | 0.41% / 0.49% / 0.06% |

The falsifiable prediction committed blind in the Day 1 journal — that the 49,152-token point
degrades *less* than the 512-token point on the worker victim, as the handoff-rate mechanism
requires — held: 0.06% vs 0.41%.

**Even load distribution.** During the Week 13 bring-up, a concurrent probe of the worker pool
landed all eight in-flight requests on the same worker — an 8/0 split — despite both workers
being live and individually routable. The diagnosis at the time — later undercut by
measurement, below — was that the nginx config lacked a shared-memory `zone`, so under
`worker_processes auto` each nginx worker process kept its own private `least_conn` connection
counters and, cold-starting from zero, defaulted to the first-listed backend. The claim that
the front door actually balances was therefore held open until a re-probe showed real
distribution. It now has: during the Week 15 R2 flood, on the v0.23.0 boot path, the pool split
680/663 (50.6/49.4) across the two workers.

Attribution honesty: the evenness is **not** credited to the nginx `zone` fix. Week 14
Session 4 established that the zone directive — while documentation-correct for stateful
balancing under `worker_processes auto` — has an *unobservable* effect on this symmetric
two-server pool: distribution measured even with and without it. The zone is retained as
defensive correctness; the even split is a property of `least_conn` on symmetric backends under
this traffic.

### Mechanism

With disjoint GPUs, interference can only flow through shared host resources. The observed
coupling channel is **host CPU contention, userspace-dominated and loaded-not-exhausted**:
under R2/R3 load, host busy rises from ~1% idle to 45–50%, decomposing ~85% user / ~15% system —
the runtimes' own per-token userspace work (scheduling loop, tokenize/detokenize, sampling
dispatch, SSE streaming) contending across the shared six cores; kernel (system) time is the
smaller component of the rise, at roughly one sixth of the userspace share. No logical CPU
pegged (max 71.6% in R2, 80.2% in R3); RAM flat throughout (~40.8 GiB available, swing
≤0.06 GiB). The load level and its decomposition are measured; the causal reading — the shared
resource is loaded but never exhausted, so victim per-token scheduling is never starved, which
is *why* the tiers stay isolated — is interpretation, retained after the alternatives below
were considered and disposed of.

Two channels hypothesized at design time were ruled out or set aside:

- **PCIe byte volume: not a channel.** Steady-state serving moves token IDs, not tensors — a
  50K-token prompt is ~200 KB, sub-millisecond even on a Gen4 x1 link. Link width governs
  model-*load* time, not serving. A design-time argument, unchanged by the measurements.
- **Interrupt rate: unsupported-and-unnecessary.** Hardirq time measured 0.000% across all 12
  logical CPUs, idle and loaded (softirq ~0.005–0.03%, confirming the pipeline resolves per-mode
  CPU correctly). This disposes of interrupt-*servicing cost* as a channel; it cannot measure
  interrupt *rate*, and near-zero servicing time is the expected reading for MSI-X (Message
  Signaled Interrupts — the modern PCIe interrupt-delivery mechanism, whose per-interrupt
  handler cost is microseconds) even under heavy interrupt load. The disposition, recorded in
  the Week 15 hardirq resolution, is therefore "unsupported and unnecessary" — host-CPU
  contention already fully accounts for the observed ≤0.5% coupling — not "proven absent."

The finer signature — worker-victim decode degradation concentrated at short context and
decaying toward the ceiling (0.41% → 0.06%), as a handoff-rate mechanism predicts — is
**consistent with** that mechanism but not a clean measurement of it: the 0.41% signal sits
≈1× the worker's decode run-to-run spread (≤0.38%). It is reported because the decay shape
matched the blind Day 1 prediction; it is hedged because the signal sits at the noise floor.
The isolation verdict does not depend on this reading.

---

## Scope and honesty constraints

These are load-bearing for how the numbers above may be read:

1. **49,152 is a large-context point, not near-ceiling.** It is ~25% of the orchestrator's
   empirical 193,837-token ceiling and ~13% of the worker's — a relabel recorded in the Week 15
   resolution.
2. **Deep-context isolation (150K+) is extrapolated, not measured.** The extrapolation is safe
   under the record — decode degradation decays monotonically out to 49K and the rate mechanism
   predicts continued decay, so the deep-context regime can only be more isolated — but it is an
   extrapolation and is stated as one.
3. **These numbers are a terminal characterization of the 4× RTX 3090 topology.** The coupling
   mechanism they characterize — multiple vLLM processes contending on shared host CPU across
   physically separate GPUs — does not exist in the same form on a single-die successor
   platform, so the values must not be trended forward. What transfers is the thesis, the
   method (committed predictions, pre-registered isolation bars, saturation gates, empirical
   placement verification), and the demonstrated *pattern* of operational proof — not the
   numbers.
4. **The 131K–262K worker context range remains quality-unvalidated** (memory fits it; the
   validation boundary does not cover it). Open item, carried from Week 12.

---

## Closing the arc

Week 11 ended with a negative result: neither single configuration serves the interactive
use case. Weeks 12–13 built and quality-validated the two-tier answer; Week 14 put the
toolchain, the repos, and the front door in order; Week 15 supplied the one claim the
architecture still owed itself — that the tiers stay isolated *while running concurrently
under load*. With that measured, the delegation architecture is validated end to end on this
substrate: viable (W12), quality-lossless under quantization (W13), co-resident (W13), evenly
fronted (W15), and interference-isolated (W15).

The thesis stands independent of the substrate that proved it.

---

## Sources

- Week 11 summary — `phase-3-…/week-11/week-11-summary-fp8-31b-parallelism-context-ceilings-journal.md`
- Week 11 Day 5 — TP=2 under load — `…/week-11-day5-tp2-wins-under-load-journal.md`
- Week 12 summary — 12B-QAT sub-agent tier — `…/week-12-summary-12b-qat-sub-agent-tier-journal.md`
- Week 13 Day 2 — boot choreography + nginx front door; Day 5 — native full-stack boot;
  Days 8–9 — QAT-vs-BF16 quality (both tiers)
- Week 14 Session 4 — nginx `least_conn` zone null result — `…/week-14-session4-nginx-leastconn-zone-null-result-journal.md`
- Week 15 Day 1 — design + committed predictions; Day 2 — execution, verdict, and the appended
  hardirq resolution (R `825b3c8`, `8617dbe`; tool SHA `T@88493e1`)
- IRS — `nginx/nginx.conf` (routing contract, zone commentary)
