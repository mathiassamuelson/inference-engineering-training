# Week 16 — Session 3: the program capstone

**Dates:** 2026-07-09 (gates, grounding, claim inventory, draft, first review rounds) →
2026-07-14 (fresh-eyes review, second-reviewer findings, commit). One session across
multiple sittings.
**Session type:** writing/synthesis (Claude Code). No measurement; nothing historical
rewritten.
**Deliverable:** `docs/program-capstone.md` (commit `96db8d6`), with its committed
claim→source map at
`phase-3-optimization-and-quantization/week-16/week-16-session3-capstone-claim-inventory.md`
(`7e0cf03`, appended corrections `265afa3`, `aabaea0`).

---

## Scope

Third session of the Week 16 conclusion sequence (renames → consolidation → **capstone** →
method Pulse → plan closure). One document: the program's spine for a cold public reader,
Week 1's ~5,000 tok/s transformers plateau to the quality-validated, concurrency-proven
two-tier delegation architecture on pinned v0.23.0. The Phase 3 summary does not exist yet
(Week 16 is open), so the capstone's Phase 3 material cites the Week 11–15 weekly summaries
directly — the agreed sequencing inversion; consistency with the future phase summary is by
construction, since both trace to the same dailies.

**OUT:** the method Pulse (Session 4); plan closure, the Week 16 summary, and the Phase 3
summary (Session 5); the successor program; any measurement.

## Pre-session gates

`git pull --rebase` (R, T) + `--ff-only` (IRS): all clean and current. Session 2's seven
summaries and its journal (with both appended corrections) confirmed on origin.

## Protocol — the strictest version, and a new standing model executed for the first time

Two stops, per the pickup:

1. **Inventory stop, on disk.** The grounding set was read first (phase 1/2 summaries, weekly
   summaries 8–15 and week reports 1–7, `delegation-architecture.md`, `training-plan.md`;
   dailies for spot-verification). The tagged claim inventory (~70 claims in seven sections,
   every claim with a document-and-section source reference) was written to the repo tree and
   reviewed in an editor. Two claims were added at review: **G5** (task-fit over headline
   capability as a program through-line — with guardrails written into the claim: not dated
   to a moment of emergence, and not "smaller matched bigger," which was never measured) and
   a candid retelling of **D22** (the Week 15 "near-ceiling" error told as what it was:
   half a dozen ceiling-adjacent numbers in circulation and the design anchored to the
   retired FP8 one).
2. **Inventory committed before drafting** (`7e0cf03`) — the first full execution of the
   commit-the-inventory-pre-draft standing model, mirroring commit-predictions-before-
   measurement. Post-commit inventory changes were appended corrections, per the model:
   `265afa3` (B4 scope-tightening, below) and `aabaea0` (D15a addition, below).
3. **Draft stop.** The draft went through multiple review rounds (below) before commit.

## The review rounds — what they caught

The draft review was the longest of the program's derived-document reviews, and its catches
sort into three classes worth recording:

**Register catches (the contract working):**

- **B4 over-explanation.** A legibility expansion of the Week 2 TensorRT claim asserted a
  general root cause (fixed-shape vs dynamic-shape models) and an unmeasured background claim
  ("reliably speeds up image classifiers"). Neither is in the record — the measured slowdown
  was pinned to device placement, the dynamic-ops attribution applies only to the
  direct-export trace failure, and no vision model was ever measured. Ruled
  unsupported-by-the-record, removed; the capstone now marks the counterfactual (whether a
  correctly placed engine would have won) as uninvestigated. Appended to the inventory as a
  correction. The general lesson: glosses that define *terms* are safe; glosses that supply
  *causes* need record support.
- **Use-case scoping restored.** The quality claims ("quality-lossless") were unscoped in the
  draft; the record scopes them to the use case's own probes and rubrics. All four instances
  now carry "for this use case," and the Week 13 passage states why the scoping is the
  task-fit principle applied to evaluation.
- **D15a added.** The worker-tier deployment decision (two independent workers over TP=2,
  despite TP=2's per-model win) was argued at review and added to both inventory and
  capstone — with the two-workers-beat-TP=2 comparison tagged interpreted-from-measured
  (derived arithmetic; the Week 14 journal never draws it), the single-NVLink-bridge
  commitment as record, and the second-bridge rationale marked author-supplied at review.

**Legibility catches (a systematic sweep, prompted by one example each):**

- Every ratio and percentage got its referent ("1.56× *what*"); the ~20× cost-per-token
  figure got its arithmetic shown after the reviewer's own mental math produced ~9.8×.
- Every domain concept got a first-use gloss (parallelism strategies, KV cache,
  prefill/decode, MoE, QAT, LLM-as-judge, least_conn, hardirq/softirq, digest pinning); the
  sweep's largest find was structural — **the driving use case was never defined anywhere**,
  and is now introduced at the Phase 3 opening.
- Every model reference names its model canonically; journal-register vocabulary that leaked
  into prose was rewritten ("binary-searched," "version convergence," "the week paused
  itself," inventory-style parentheticals); two `$` signs that triggered LaTeX rendering in
  Markdown viewers were spelled out; overlong sentences split; the whole file rewrapped to
  ~97 columns (content-verified by a full end-to-end re-read).

**Second-reviewer findings (a parallel review, relayed 07-14):**

1. **Blocking, and correct:** the capstone re-asserted Week 11's "neither config suffices" as
   the delegation motivation without saying what the architecture doc says prominently — the
   QAT migration lifted that forcing constraint. This was inventoried as D26 and never
   rendered in the draft. Fixed with a paragraph after Week 13: the ceiling no longer binds;
   the architecture stands on context-management/token-cost and concurrency-at-depth; the
   re-examination lives in the architecture doc.
2. The 131,072 hedge now matches the architecture doc ("validation boundary *at the time of
   the Week 12 validation*").
3. The Week 15 verdict now uses the plain-language headline the architecture doc adopted
   ("neither tier slows the other down") instead of the undefined "both tiers isolated."
4. The four-Pulse count was verified against `docs/linkedin/README.md` (four entries — the
   count is right).
5. Hard-wrap artifacts normalized (the rewrap above).

**Additions at review:** a **model index** table (every model, its quantization variants,
role, and where it appears) — motivated partly by the document's own Week 15 lesson about
variant confusion.

## Deliverables / state

- `docs/program-capstone.md` — committed `96db8d6`, pushed.
- Claim inventory with two appended entries — committed `7e0cf03` / `265afa3` / `aabaea0`,
  pushed.
- This journal.
- R / T / IRS clean at close.

## Carried forward

1. **Session 4: the method Pulse** — structured AI-assisted self-training as a method; the
   paper trail as evidence. One candidate thread noted during this session's G5 discussion:
   the first-person "where the task-fit thinking emerged" reflection belongs there, not in
   the capstone.
2. **Session 5: plan closure** — final Key Changes entries and footer status in
   `training-plan.md`, the Week 16 weekly summary, and the Phase 3 summary.
