# Phase 3 Summary — Optimization & Quantization: The Delegation Arc (Weeks 11–16)

**Span:** 2026-05-30 → 2026-07-18 (Week 11: May 30 – Jun 9; Week 12: Jun 9–12; Week 13: Jun 13
onward; Week 14: late June; Week 15: early July; Week 16: Jul 8/9 – Jul 18)
**Hardware:** 4× RTX 3090 throughout — GPUs 0+2 on the AORUS NVLink bridge (NV4), GPUs 1+3 on
PCIe 3.0 x1; the geometry Phase 2 handed over, used exactly as handed
**Model arc:** Gemma 4 31B Dense (FP8 → **QAT W4A16** at Week 13) as the orchestrator; Gemma 4
12B (**QAT W4A16**) as the worker tier; both BF16 parents as the Week 13 quality references
**Stack arc:** vLLM 0.21.0 (pinned digest) → the Week 12 `gemma4-unified` scaffolding →
**v0.23.0** (digest `sha256:6d8429e3…22ed8f`), the converged image on which every workaround
retired and the production stack froze; nginx `least_conn` front door from Week 13
**Sources:** the six weekly summaries (`w11`–`w16-summary`), `docs/delegation-architecture.md`,
`docs/training-plan.md` §Phase 3 and §Key Changes. Per-claim register detail lives in the
weekly summaries; this document defers to them, as the Phase 1/2 summaries do.

## TL;DR

Phase 3 was planned as the optimization-and-quantization phase and became the **delegation
arc**: the six weeks in which one measurement reframed the program's goal, an architecture was
designed to that reframe, validated tier by tier, proven operationally, and then written down —
concluding the program. Week 11 closed the long-running parallelism thread and produced the
phase's pivotal finding: on this hardware, **no single serving configuration of the 31B serves
the interactive use case** — tensor parallelism is interactive but context-limited, pipeline
parallelism has the context but is not interactive; *fit is not the bar, usable-for-the-task
is*. Week 12 validated the answer's cheap half (a 12B QAT worker on a single 24 GB card). Week
13 validated its quality (QAT W4A16 ≡ BF16 parent at **both** tiers, by position-bias-controlled
LLM-as-judge evaluation), converged the whole stack onto one pinned image with zero per-model
workarounds, and brought all three services up concurrently. Week 14 was the close-out that
paid the accumulated debts — the toolchain/results repo split, the 12B parallelism
characterization, the throughput Pulse, and the phase's flagship null result (the nginx
`least_conn` zone fix changed nothing measurable). Week 15 delivered the operational proof:
under a committed prediction table with a falsifiable mechanism claim, **neither tier slows the
other down** at any measured size, and the delegation-architecture write-up completed. Week 16
concluded the program — renames, the consolidation layer, the capstone, the public method
article, and plan closure — with zero measurements, by design.

The phase's quantization stance, set in Week 12–13 and held: **highest-fidelity-that-fits**,
not quant-versus-quant benchmarking. QAT W4A16 earned the production slot by fitting the
hardware, matching its BF16 parent on quality, and outrunning FP8 on decode by +36–50% across
the prompt-size ladder — the "smaller AND faster" hypothesis, committed with its mechanism
before the sweep ran (decode is bandwidth-bound; w4a16 moves half of FP8's weight bytes per
token) and confirmed in both sign and slope.

## Week 11 — the parallelism close and the reframe

The closing chapter of the thread running Week 3 → 7 → 8 → 9 (paused on #39133), unblocked by
0.21.0's HMA fix. On the 31B FP8: a two-coefficient KV cost model (≈ 1.97 GiB + 39.2 KiB/token
per sequence per GPU, TP=2) validated to ~1% and used to predict every subsequent ceiling
within 1.5%; text-only serving reclaiming the vision tower's ~1.1 GiB/GPU for KV; PP=2
**non-viable** (the 256K-vocab embedding/LM head land whole on end stages, starving KV ~12×
vs TP=2); PP=4 viable but ~1.7× slower on decode, structurally, with placement steering
provably irrelevant; and under c=4 load, TP=2 beating PP=4 on aggregate throughput *and*
fan-out completion at every prompt size — the bigger-pool-means-more-throughput thesis
refuted; capacity and throughput are distinct ceilings.

Day 6 found the ceilings — TP=2 KV-bound at 54,496 tokens (util 0.95), 66,848 with the
CUDA-graph tax recovered (util 0.97, +22.7%); PP=4 architecture-bound at the full 262,144 but
serving it at ~15 tok/s with ~5-minute TTFT — and with them, the reframe: the first read
scored PP=4 the winner on fit, and fit was the wrong bar. **Neither config suffices** for an
interactive operator loop. That result is what motivated the two-tier delegation architecture
(31B orchestrator at its interactive ceiling; small workers fanning out bulk-context reads),
and everything after Week 11 is that architecture being built and tested.

## Week 12 — the worker tier validated

The 12B QAT checkpoint loads and serves on a single 24 GB card (8.28 GiB weights); Day 1's
OOM was self-inflicted (a shallow-replacing `--hf-overrides`), plus one genuine image bug
patched via a three-line upstream backport, both retired at version convergence. No memory
ceiling exists on the card — the full 262,144 architectural context fits at 2.16×
concurrency. Production MML pinned to **131,072** — the model's `max_position_embeddings`
pin, read as its quality-validation boundary; the 131K–262K range fits in memory but is
quality-unvalidated — an open item the phase carried to its close and hands onward. Single-GPU
serving has no inter-GPU traffic, so the PCIe-x1 link never enters the worker's critical path
— the slots Week 11 wrote off for every parallel topology are exactly right for single-card
workers, which is what makes the four-GPU geometry work at all. Measured single-card throughput: decode 69.6 tok/s
@8K / 51.7 @64K / 46.2 @102K; batching pays 2.33× at 8K but the worker is functionally serial
at 64K+ — a direct input to the front-door design (at depth, queueing ≈ batching, with better
latency). Verdict: **go** on the delegation architecture.

## Week 13 — quality, convergence, co-residency

The week's center of gravity became the quality question, pulling a focused slice of the
originally-planned quantization-quality work forward: **is QAT W4A16 lossless against its
BF16 parent for this use case?** Answer, both tiers: yes, within the instruments' power to
detect. Orchestrator: guardrail adherence an 8/8 tie, decisive pairwise verdicts split
evenly, no order-robust regression on any axis. Workers (two components, purpose-built
extraction contract): format 6/6 strict-conformant on both models, pairwise parity once
position-sensitivity is discounted, pointwise 4.83–5.0 — good enough in absolute terms, not
merely no-worse. Method: matched-provenance captures, task-appropriate rubrics, both-orders
position-bias control, and deterministic format checking kept separate from the LLM judge.

The same week converged the entire stack onto `vllm/vllm-openai:v0.23.0` (pinned digest) —
all three production models plus both BF16 parents loading with **zero** per-model
workarounds, retiring every piece of Week 12 scaffolding — and demonstrated three-service
co-residency on the one host (~40 GB of 64 GB RAM to spare; simultaneous boot ~2.5× faster
than staggered). QAT-vs-FP8 throughput landed here too: decode +36–50%, prefill +1.8–3.9%.
What Week 13 could not yet claim was isolation under load; that became Week 15's question.

## Week 14 — the close-out week

Inserted by the phase's own accumulated loose ends, and worth its week. The
**toolchain/results repo split** (T carved out of R) dissolved the dirty-tree provenance
problem at its root rather than classifying around it — `provenance.py` anchoring the
recorded SHA to the tool repo via `__file__`, so a capture run from R records T's commit.
The **12B-QAT parallelism sweep** answered the worker tier's own layout question: TP=2 on
the NVLink pair beats TP=1, so the worker model benefits from the pair when it's free — an
option the production layout doesn't use (the pair belongs to the orchestrator) but the
record now holds. The **nginx `least_conn` shared-zone fix** produced the phase's flagship
null result: introduced expecting it to matter under concurrent load, measured properly, it
changed nothing distinguishable from the even split already in place — kept in the record by
name, with both halves of the wrong prediction preserved. The held throughput Pulse
published. The repo reorg consolidated weekly reports into the phase/week tree and set the
bare-`week-NN` go-forward convention.

## Week 15 — the operational proof

The one measurement still owed: does the two-tier layout actually stay isolated under load,
given shared host CPU, PCIe root complex, and RAM? A committed prediction table per tier and
regime — including one deliberately falsifiable claim (the interference signature, if real,
had to concentrate at short context and decay toward the ceiling, or the proposed mechanism
was wrong) — then solo baselines on the frozen stack and two loaded regimes at three prompt
sizes, against a pre-committed isolation bar of ≤3% decode *and* prefill degradation with a
≥95% aggressor-utilization gate on every run. **Verdict: neither tier slows the other down.**
All five predictions held, the falsifiable one included; the scorecard still records where
observations deviated from predicted bands, because a win scored carelessly teaches nothing.

The delegation-architecture write-up completed the same week, folding in the concurrency
evidence — and restructured constraints-first after review caught the phase's most
instructive self-correction: the "neither config serves" forcing constraint had been
measured on the **FP8** orchestrator, and the QAT migration roughly tripled the TP=2 KV
budget (empirical ceiling walked to 193,837 tokens at the production operating point), so a
single QAT orchestrator now serves the validated 131K envelope interactively. The FP8
numbers stand as the historical record of what forced the decision; the architecture stands
on the two constraints that survive it — context-management/token-cost, and
concurrency-at-depth. The near-ceiling anchor error that surfaced during design (half a
dozen ceiling-adjacent numbers in circulation, the design anchored to the retired FP8 one)
is told candidly in the capstone's inventory and the Week 15 record.

## Week 16 — the program conclusion

Five sessions, zero measurements: repo renames first so every closing document carries final
names; the consolidation layer (seven weekly summaries, the Phase 1 and 2 summaries); the
capstone (`docs/program-capstone.md`) as the first full execution of the
commit-the-inventory-before-drafting model; the method Pulse (published 2026-07-16) with its
mid-session amendment adding `docs/training-method.md`; and plan closure — the final Key
Changes row, the concluded footer, this summary. Detail in `week-16-summary-…`.

## Supersessions and corrections across the phase

- **Week 11's "neither config suffices" → forcing constraint lifted by the Week 13 QAT
  migration.** The finding was true of the FP8 orchestrator it was measured on; QAT's KV
  budget removed the ceiling as a binding constraint. The architecture was *not* thereby
  invalidated — it stands on token-cost and concurrency grounds — but the record now says
  which argument is historical and which is live (`docs/delegation-architecture.md` carries
  the re-examination; the capstone was corrected at review to match).
- **Week 13's missing-`zone` diagnosis for nginx imbalance → not confirmed.** The Week 14
  zone fix was a null result; the even upstream split is credited to `least_conn` on
  symmetric backends, and the Week 13 diagnosis is preserved as the hypothesis it was.
- **The Week 15 near-ceiling anchor error** — the interference design initially anchored
  "near-ceiling" to a retired FP8 ceiling figure; caught before it could void the finding,
  and 49,152 is labeled throughout as a large-context point (~25% of the ceiling), not
  near-ceiling.
- **Version convergence retired the Week 12 scaffolding** — the source-patched launcher,
  `start-12b-qat.sh`, and the `--hf-overrides` workarounds are historical artifacts only;
  v0.23.0 loads everything clean.
- **The plan's own Phase 3 was rewritten by the work three times** (quantization weeks →
  delegation weeks; the close-out week inserted; the program concluded at 16) — every change
  a dated Key Changes row, none a rewrite.

## Not measured in this phase (as the record leaves it)

- **True near-ceiling interference isolation (150K+ context)** — extrapolated from monotone
  decay, stated as extrapolation everywhere it appears.
- **The interrupt/DMA-completion rate hypothesis** — disposed of as unsupported and
  unnecessary, not refuted; the counters used cannot measure it.
- **Worker quality in the 131K–262K context range** — fits in memory, quality-unvalidated,
  open since Week 12.
- **The broad quantization quality-degradation curve** — deliberately migrated to the
  successor program rather than run here; Week 13's QAT-vs-parent equivalence is the focused
  slice the phase pulled forward.
- **Speculative decoding, KV-cache compression, NSight profiling** — displaced by the
  delegation arc and the conclusion; deferred as topics of continued interest.

## What Phase 3 concluded with

- **The production reference:** the frozen two-tier stack on v0.23.0 — 31B-QAT orchestrator
  (TP=2, NVLink pair) + 2× 12B-QAT workers (TP=1, x1 cards) behind nginx `least_conn` —
  quality-validated, concurrency-proven, and closed as the program's experimental reference.
- **The documents:** `docs/delegation-architecture.md`, `docs/program-capstone.md`,
  `docs/training-method.md`, the published Pulse series, and the consolidated
  weekly/phase-summary layer this document completes.
- **The toolchain (T)** in its final form: provenance anchored to the tool repo, the judge
  and probe harnesses, the bring-up gates, the launchers with role presets.
- **The method, exercised at full strictness:** committed predictions with a falsifiable
  mechanism claim (Week 15) and committed claim inventories before prose (Week 16) — the
  same discipline pointed at measurement and at writing.
- **The handoffs:** the migrated work and open items above, to the successor program's own
  plan — which this document, like the capstone, makes no claims about.

## Register notes

- This summary consolidates the six weekly summaries and defers to them for per-claim
  register detail (prediction misses, noise bands, confound inventories), per the Phase 1/2
  precedent. The capstone's claim inventory (Section D) is the claim-by-claim map for the
  Weeks 11–15 material and was written against the same weekly summaries.
- Written **after** the capstone, per the Week 16 sequencing inversion; consistency is by
  construction — both documents trace to the same weekly summaries and dailies.
- Week 12 facts verified against `week-12-summary-12b-qat-sub-agent-tier-journal.md` directly
  (initially drafted from the plan's outcome record and the architecture doc; the verification
  pass confirmed all figures and sharpened the MML-131,072 phrasing to the summary's own
  hedged register).
- All quoted figures are measured values from the weekly summaries; the Week 15 verdict is
  stated in the write-up's adopted plain-language form ("neither tier slows the other down")
  with the pre-committed ≤3% bar named rather than re-adjudicated here.
- Week 16 contains no measurements; its section is record-class throughout.
