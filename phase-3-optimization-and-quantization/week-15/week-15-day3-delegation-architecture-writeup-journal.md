# Week 15 — Day 3: delegation-architecture write-up, finalized

**Date:** 2026-07-08
**Session type:** writing/reasoning (web chat). No commands, no probes, no measurement — the
data was committed in Sessions 1–2 and cited, not re-argued.
**Deliverable:** `docs/delegation-architecture.md` (final) — the second and last Week 15
deliverable. **This closes Week 15.**

---

## Scope (declared at open)

- **IN:** finalize the delegation-architecture write-up under the full confidence-provenance
  contract (derived, results-bearing document); fold in the Week 15 operational proof; close
  the arc opened by Week 11's "neither config serves the use case."
- **OUT:** Week 16; the successor program; any new measurement; the prior-week Pulse and
  parallelism-sweep deliverables.

## Pre-session state

Gates confirmed clean (R/T/IRS pulled, trees clean; Week 15 interference artifacts pushed:
R `825b3c8`/`8617dbe`, tool SHA `T@88493e1`).

## Framing decisions (settled before the inventory)

1. **Use-case naming:** *operator copilot* throughout; the superseded project name is not
   mentioned anywhere in the document.
2. **Structure:** thesis-led. The thesis (capable orchestrator + cheap workers,
   substrate-neutral) opens the document; the consumer-GPU system is presented as the proof
   case, not the thesis.
3. **Register:** mixed tags in the review draft — `settled`/`argued` for design claims,
   the results register (`measured`/`interpreted`/`not-measured-here`/`open`) for evidence
   claims — matching the Day 1 journal's precedent.

## Process

Claim inventory first (15 claims, tagged), reviewed and confirmed before any prose. Review
draft written with inline tags; iterated through a full review pass; tags stripped at final
with `argued` tags converted to one-clause prose traces, and `assumed`/`open` claims left
explicitly marked in prose.

## The substantive review findings (worth more than the copyedits)

1. **The motivating argument's ceiling leg did not survive the QAT migration — and the
   review caught the draft resting on it.** The Week 11 "neither config serves the use case"
   measurement was made on the FP8 orchestrator; the Week 13 QAT migration roughly tripled the
   TP=2 KV budget (empirical ceiling 193,837 tokens at the production operating point; 262,144
   refused, est. max ~218,624), so a single QAT orchestrator serves the validated 131K envelope
   interactively. The section was restructured **constraints-first**: (1) context management and
   token cost, (2) context ceiling / interactivity — the design-time forcing constraint, later
   lifted, (3) concurrency at depth. The FP8 numbers now stand as the historical record of what
   forced the decision, not as the document's live argument.
2. **Constraint 3 (concurrency at depth) is labeled as discovered in testing**, not as a
   design-time premise — traced to the Week 11 under-load finding (orchestrator blocks on the
   slowest request in a wave) and the Week 12 worker seriality at 64K+.
3. **The counterfactual is marked as such:** had the program started on the 31B QAT, the
   ceiling would not have been binding, but we believe the same decision would have followed
   from the other two constraints — stated in the document as a judgment, not a derivation from
   the record.
4. **Lifecycle asymmetry added** to the concurrency argument: long-lived orchestrator
   accumulating investigation state vs. ephemeral per-task workers (design property, not a
   measurement).
5. **Zone-attribution honesty held.** The even 680/663 split is credited to `least_conn` on
   symmetric backends, not to the Week 14 `zone` fix (null result); the Week 13 missing-zone
   diagnosis is presented as the hypothesis it was, later undercut by measurement.
6. **Util-asymmetry rationale recorded** (a reader question the draft hadn't answered): workers
   at 0.90 because the pool already exceeds any need (full 262K at 2.16×); orchestrator at 0.95
   because 0.90 refused to boot during the Week 13 bring-up under the CUDA-graph reservation.
7. **One claim deliberately withheld:** the recollection that the 12B's 131,072
   `max_position_embeddings` pin was later recognized upstream as a config bug and fixed is
   **not in the committed record** — the document carries only the temporal hedge ("validation
   boundary at the time of the Week 12 validation"). If a citation (upstream issue/commit)
   turns up, the statement can be strengthened; until then it would be
   unsupported-by-the-record.

Legibility fixes from the same pass: TP/PP and MSI-X glossed on first use; victim/aggressor
defined; the "isolated" verdict restated in plain language and anchored to the pre-committed
3% bar; several journal-shorthand idioms ("pays", "minor partner", "follows the hardware's
grain", "three configurations that each boot") rewritten as literal prose.

## Honesty constraints carried into the final (verified present)

- 49,152 labeled a large-context point (~25% of the 31B ceiling), not near-ceiling.
- Deep-context isolation (150K+) stated as extrapolation, safe by monotone decay.
- All interference numbers scoped as a terminal characterization of the 4×3090 topology; no
  forward claims onto the successor platform — what transfers is thesis, method, and pattern.
- The decay signature stated as consistent-with the rate mechanism, at ~1× decode noise.
- Interrupt channel: "unsupported and unnecessary," not "proven absent."
- 131K–262K worker range: open, carried from Week 12.

## Deliverables / state

- `docs/delegation-architecture.md` — finalized (review draft with inline tags → tag-stripped
  final), committed to R this session.
- This journal: `week-15-day3-delegation-architecture-writeup-journal.md`.
- **Week 15 closed** — both deliverables done (interference characterization, Sessions 1–2;
  architecture write-up, this session).

## Carry-forwards (not this session)

- **Tool fix:** `start-stack.sh` stale `MODEL_31B`/`MML_31B` default (carried from Session 2).
- **Optional:** source the upstream fix for the 12B 131,072 MML pin, if it exists, and
  strengthen the write-up's hedged sentence with a citation.
- **Week 16** — program conclusion (renames → journal consolidation → capstone → method Pulse).
  Separate week, separate chat.
