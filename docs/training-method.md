# The Training Method

This document states the working method that governed the inference-engineering-training
program and traces each rule to the moment in the record that forged it. It is a companion to
the program capstone, which owns the technical findings, and to the public article on
AI-assisted self-training, which owns the partnership story and links here. Nothing in this
document is a result; everything in it is a rule, and every rule has a citation.

*Produced under the Session 4 claim inventory
(`phase-3-optimization-and-quantization/week-16/week-16-session4-method-pulse-claim-inventory.md`),
per the claim-inventory contract described below.*

## The method, compactly

- **Predict before measuring, in writing, then score the prediction.** Committed predictions
  are never rewritten; corrections are appended.
- **One experiment at a time**, with explicit in/out scope declared at session open.
- **A result that shows no effect is still a finding** — kept in the record by name, never
  quietly dropped.
- **Journals are never rewritten**; corrections are appended, dated, below the original text.
- **Don't build on a known bug** — halt rather than measure on a broken foundation.
- **What we learned got written down with the same care as what we measured** — study
  documents carry the same rigor as results.
- **Everything traces.** Every claim in a derived document traces to the daily journals;
  evidential tags are applied before prose; a claim inventory is committed before drafting
  begins. The appended corrections cited below are proof the mechanism catches errors — not a
  claim that none occur.

None of these rules came from a textbook on learning. Each was forged by a specific moment in
the program — usually a moment where we nearly got something wrong.

## Predict before measuring, in writing, then score it

Predictions in this program were not guesses noted in passing. Before a measurement ran, the
prediction was committed: expected outcome, the mechanism behind it, and a stated confidence.
A representative instance is the KV-memory prediction set in
`week-11-day2-fp8-31b-tp2-kv-characterization-journal.md` (§KV Characterization), committed
before the measurements it predicts.

The rule grew teeth in Week 15. Before the program's final characterization — cross-tier
interference between models sharing the host — we committed a full prediction table: per
experimental regime, a predicted outcome with mechanism and confidence
(`week-15-day1-interference-design-predictions-journal.md`). One prediction was deliberately
falsifiable: the interference signature, if real, had to concentrate at short context and
decay toward the near-ceiling regime. We declared in advance that the reverse shape would
mean our proposed mechanism was wrong — not incomplete, wrong.

All five predictions held, including the falsifiable one. The scorecard
(`week-15-day2-cross-tier-interference-isolated-cpu-handoff-journal.md`, §Prediction
scorecard) nonetheless records where observations deviated from the predicted bands — one
nominal miss sat within baseline noise and was scored as such, with the noise comparison
shown; one observed value fell slightly *below* its predicted band and the scorecard says so,
on a prediction that held. Recording the deviations on held predictions is what keeps a
scorecard honest — a win scored carelessly teaches nothing.

## One experiment at a time

Forged by a near-miss, recorded in `week-11-day4-pp4-viable-vs-tp2-journal.md`. During Week
11's placement work, the first decode timing on each server right after model load
consistently came back at roughly 9 seconds for the probe, while every subsequent probe on
the same warm server returned roughly 5. Treated as single samples — which is what you take
when you are moving too quickly — the data flipped the conclusion between "placement is
irrelevant" and "the interconnect halves decode latency." Both readings were noise.

What caught it was slowing down: lining the probes up by ordinality rather than by placement,
which exposed a one-time cold-start cost on the first decode request after load — repeatable
at the same magnitude on both placements, and irrelevant to the question being asked. Two
wrong conclusions, nearly recorded as findings, from one warm-up quirk. One controlled
comparison at a time is not a productivity preference; it is how you notice what your data is
actually telling you.

## A result that shows no effect is still a finding

In Week 14 we introduced connection-aware load balancing across the worker tier — nginx
`least_conn` with a shared state zone — expecting it to matter under concurrent load. It was
measured properly. It did not matter: no effect distinguishable from the even split already
in place. The result is in the record by name, as a finding
(`week-14-session4-nginx-leastconn-zone-null-result-journal.md`).

A program that records only its successes teaches a fiction. The experiments that show no
effect are where a mental model gets corrected, and a no-effect result that goes unwritten is
a correction that never happens.

## Journals are never rewritten

Daily journals are the program's primary record, and the rule is absolute: they are never
edited after the fact. When something in a journal proves wrong, a dated correction is
appended below it; the mistake stays visible and the fix stays attached to it.

The record contains worked examples. The Week 16 consolidation work appended two corrections
to its own session journal rather than amending the text they correct
(`week-16-session2-journal-consolidation-journal.md`). The capstone's claim inventory carries
two appended corrections of its own, as commits below the committed original (corrections
`265afa3` and `aabaea0`, appended to inventory commit `7e0cf03`). A paper trail with visible
corrections is evidence that checking happens. A spotless one is evidence of nothing.

## Don't build on a known bug

In Week 9, a reproduction of KV-cache sizing behavior surfaced an underlying issue in the
serving stack (`week-09-day3-gemma4-kv-sizing-reproduction-journal.md`) — and the program
stopped. Not the affected experiment: the program. Measurements halted and progress paused
until the bug was actually resolved.

It cost real calendar time, and it was the right call for the same reason the cold-start rule
is: results accumulated on a foundation known to be broken are not results. They are future
corrections, chosen in advance.

## What we learned got written down with the same care as what we measured

The same discipline applies to learning documents. Midway through the program we decomposed
the transformer feed-forward network — from raw matrix mechanics up through why the
parallelization strategy we had measured behaves as it does — and committed it as the first
entry of a reference series
(`docs/compendiums/entry-01-ffn-inference-shapes-tensor-parallelism.md`). Its preamble flags
which claims are convention- or implementation-dependent: the evidential discipline living
inside a study document, not only inside journals.

## The provenance mechanics

Two tiers. Daily journals — the primary record — skip formality but keep doubt visible
inline: assumptions marked, no false resolution. Derived documents — plans, write-ups,
summaries, the capstone, public articles — carry evidential discipline throughout; the
strictest form of it, the full claim-inventory contract, was adopted for the program's final
derived documents (the capstone first), the way every rule in this program arrived: when the
work showed the need. Under the contract, a claim inventory listing every claim the document
will make, tagged by evidential basis
(`measured` / `interpreted` / `assumed` / `not-measured-here` / `open` /
`unsupported-by-the-record`), each with its source, is reviewed and committed to the repo
*before* drafting begins. Post-commit inventory changes are appended corrections; the final
document's text governs. Review drafts carry inline tags; tags are stripped at final. Worked
examples: the capstone inventory (`7e0cf03`, corrections `265afa3`/`aabaea0`), and the
Session 4 inventory under which this document and its companion article were produced.

The inventory rule mirrors the prediction rule on purpose. Committing predictions before
measurement and committing claims before prose are the same discipline pointed at different
targets: decide what you believe, in writing, before the work can tempt you to believe
something more convenient.
