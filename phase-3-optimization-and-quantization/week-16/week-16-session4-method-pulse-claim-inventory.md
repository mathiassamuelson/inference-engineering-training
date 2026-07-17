# Week 16 — Session 4 — Method Pulse: Claim Inventory

**Document under contract:** LinkedIn Pulse article on structured AI-assisted self-training as a
method (the program's final public artifact before closure).

**Contract status:** This inventory is committed to the repo **before** any article prose is
drafted. Post-commit changes are appended corrections only (dated, at the bottom of this file).
The published article's text governs; this inventory records what the article was licensed to
claim and on what evidence.

**Tag vocabulary:** `measured` / `interpreted` / `assumed` / `not-measured-here` / `open` /
`unsupported-by-the-record`, plus `recollection` (first-person claims resting on the author's
word rather than the repo record — spanning memory-class claims about intent, motivation, or
experience *and* hard biographical facts that are true and externally verifiable but leave no
trace in the record; the tag marks evidential basis, not epistemic softness — never presented
as record either way) and `stated-intent` (forward-looking first-person claims).

**TBV markers:** Source references marked **TBV** (to be verified) name the evidence class but
not the exact file; they must be resolved to concrete filenames/commits during review, before
this inventory is committed. Any claim whose TBV cannot be resolved is downgraded or cut.

---

## 0. Working title

- **T-1.** Title: *"Coursework Taught Me to Talk About AI. This Is How I Learned to Do It."*
  — The title's implicit claim (coursework produced conversational familiarity; the program
  produced working depth) is `recollection` + `interpreted`, supported in the article body by
  the layer-gap framing (C-4) and the program record generally. Changeable at draft time;
  change is an appended correction here.

---

## A. Personal frame — background facts

All `recollection`-class biographical facts, first-person, no repo source. Filter rule applied:
no career-transition framing anywhere in this section or the article.

- **A-1.** One year of university in Sweden directly after high school — mathematics and
  computer science, including linear algebra. Returned to university years later via evening
  and weekend classes, over several years, completing a diploma (below bachelor level; it was
  what made the later US green-card application possible). `recollection`. *Draft-time
  discretion: the green-card connection is optional color linking to A-3; include only if it
  earns its place in the opening's length budget.*
- **A-2.** Two-plus decades in internet infrastructure. From the early 2000s: sales engineer
  and solutions architect on large enterprise and carrier accounts. `recollection`.
- **A-3.** Relocated from Sweden to the US in 2011; now a US citizen. `recollection`.
- **A-4.** Moved into technical product management shortly after relocating; currently a
  principal product architect. `recollection`. *Boundary: biography ends at current role. No
  employer named in the article (A-4a: the personal-time-and-equipment disclaimer, E-1, is
  mandatory partly for this reason — attribution is one profile-click away).*
- **A-5.** Formal AI/ML education prior to the program: two Coursera/DeepLearning.AI
  specializations (Andrew Ng Machine Learning; Mathematics for Machine Learning and Data
  Science) and three 5-day Kaggle/Google intensive courses. `recollection` (externally
  verifiable via certificates; the article does not need to claim so).

## B. Personal frame — the linear algebra arc

- **B-1.** Linear algebra learned at university, then unused professionally for decades until
  ML/AI made it unavoidable. `recollection`.
- **B-2.** A deep drill-down on transformer FFN internals was conducted during the program and
  committed as a reference document. `measured` — source:
  `R/docs/compendiums/entry-01-ffn-inference-shapes-tensor-parallelism.md`
  (*Large Language Models, Decomposed*, Entry 1). The article may name and link it (repos are
  public).
- **B-3.** That work materially changed the author's relationship with linear algebra
  (adversarial → warming). `recollection` — the compendium proves the drill-down happened, not
  how the author felt. Stated in first person only.
- **B-4.** The compendium is the first entry in an intended ongoing series. `measured` — the
  document's own closing section frames Entry 2 (attention). The article states this as a fact
  about the artifact's framing only; no successor-program description (out of scope).

## C. Personal frame — motivation and trigger

Motivation entries state what the article asserts, without surrounding context, per the
standing filter. All first-person.

- **C-1.** The coursework was real and honestly completed, and it worked at what it targeted.
  `recollection`.
- **C-2.** Realization: coursework produced conversational familiarity, not working depth —
  the ability to discuss the domain, not to debug it. `recollection` + `interpreted`.
- **C-3.** Realization: the gap between those two is not closable by more coursework; it
  requires hands-on systems work with real hardware, real failure modes, real measurements.
  `recollection` + `interpreted`.
- **C-4.** Layer-gap diagnosis: the specializations covered ML fundamentals; the intensives
  covered parts of the application layer — agents, prompt engineering; nothing touched the
  model layer (transformers, GPUs, inference) until this program. `recollection` +
  `interpreted`. This is the precise version of C-2/C-3 and the payoff of the title.
- **C-5.** Decision: acquire the depth by running the learning itself as an engineered system,
  with the same rigor demanded of a production project; purchased the 4×RTX 3090 host for the
  purpose. `recollection`. Hardware facts (4×3090, consumer platform) are `measured` —
  source: program record throughout; capstone (commit `96db8d6`).
- **C-6.** Aspirational claim, direction-of-travel only: the program's goal includes moving
  from knowledge-recall toward reasoning-from-first-principles. **The article explicitly does
  NOT claim this endpoint has been reached.** `recollection` for the aspiration; the *practice*
  is evidenced by B-2 (e.g., the TP byte-budget derivation in the compendium, §6) — `measured`
  for the practice's existence. *Excluded: any chat-only reasoning episodes with no repo trace.*

## D. Method elements and their forged-by evidence

The article states the method compactly, then evidences each element with the program moment
that forged it. Method-conduct claims about the program itself trace to the record like any
other claim.

- **D-1.** **Predict before measuring, in writing, then score it — committed predictions
  never rewritten, corrections appended.** `measured` as program conduct — sources: the
  primary record directly (the journal citations below).
  Article-scope evidence (the epistemic shape only — no measurement values in the article
  body):
  - Predictions were committed with mechanism and confidence stated before any run —
    representative instance: `week-11-day2-fp8-31b-tp2-kv-characterization-journal.md`,
    §KV Characterization.
  - Week 15's committed prediction table included a **falsifiable commitment** whose failure
    was declared in advance to invalidate the proposed mechanism; all five predictions were
    scored against measurement, all held, and the scorecard still recorded where observations
    deviated from the predicted bands — honest scoring visible even on wins. Sources:
    committed table in `week-15-day1-interference-design-predictions-journal.md`; scorecard
    in `week-15-day2-cross-tier-interference-isolated-cpu-handoff-journal.md`, §Prediction
    scorecard. *Article note: "refinements logged as refinements rather than hits" and the
    existence of a prediction whose failure would have invalidated the mechanism are the
    substance that survives into the draft. Depth constraint as D-3: the mechanism itself
    (CPU-core contention, transaction rates, regime tables) stays in the record — the article
    states the epistemic shape, not the engineering.*
- **D-2.** **One experiment at a time; explicit IN/OUT scope at session open.** `measured` as
  standing conduct — source: session journals throughout. Forged-by moment: the cold-start
  artifact — source:
  `phase-3-optimization-and-quantization/week-11/week-11-day4-pp4-viable-vs-tp2-journal.md`.
  A first-probe-after-boot artifact nearly produced two wrong conclusions: the first decode
  timing on each freshly-booted server consistently returned ~9.2s/128 tokens vs. ~5.25s on
  every subsequent warm probe; treated as single samples this flipped the reading between
  "placement is irrelevant" and "NVLink halves decode latency" — all noise. Resolved by
  lining probes up by ordinality, isolating a one-time cold-start cost on the first decode
  after boot, repeatable on both placements. *Article framing note: the incident's lesson is
  measurement discipline — single samples taken while moving fast nearly became two published
  wrong conclusions; slowing to one controlled comparison at a time is what caught it. Frame
  it as that, not as scope creep.*
- **D-3.** **Honest nulls are first-class findings, kept by name.** `measured` — forged-by
  moment: the nginx null — source:
  `phase-3-optimization-and-quantization/week-14/week-14-session4-nginx-leastconn-zone-null-result-journal.md`.
  In this article the nulls are content, not caveats. *Depth constraint: the article states
  this at method-level abstraction — an infrastructure change expected to matter was measured,
  showed no effect, and was kept in the record by name as a finding — in a sentence or two,
  with the journal linked for anyone who wants the mechanism. No nginx/`least_conn`/zone
  specifics in the article body; the technical detail belongs to the record, not the Pulse.*
- **D-4.** **Journals are never rewritten; corrections are appended.** `measured` — worked
  examples: the two consolidation-journal corrections — source:
  `phase-3-optimization-and-quantization/week-16/week-16-session2-journal-consolidation-journal.md`
  (Week 16 Session 2); and the capstone claim inventory's appended
  corrections — `measured`, source: commits `265afa3`, `aabaea0` appended to inventory
  `7e0cf03`.
- **D-5.** **Don't build on a known bug — halt rather than measure on a broken foundation.**
  `measured` — forged-by moment: the Week 9 pause — source:
  `week-09-day3-gemma4-kv-sizing-reproduction-journal.md`. When the underlying issue
  surfaced, the training program itself stalled pending bug fixes rather than pressing
  forward and accumulating results on a foundation known to be wrong. *Framing constraint:
  state it as program discipline (measurements halted, progress paused), not as a publication
  decision — the article should not present the program as publication-driven.*
- **D-6.** **The claim-inventory contract: for derived documents, an approved claim inventory
  is committed before drafting begins; evidential tags applied before prose; inline tags
  stripped at final.** `measured` — sources: capstone inventory `7e0cf03` + corrections
  `265afa3`/`aabaea0`; **this document itself** (the article may note, without cuteness, that
  the article the reader is reading was produced under the same contract, and this inventory
  is in the public record).
- **D-7.** **Understanding is an artifact too** — the same rigor applied to learning documents,
  not only experiments. `measured` — source: B-2 (the compendium, whose preamble flags
  convention-dependent claims inline). `interpreted` as a named method element (the phrasing
  is the article's; the practice is the record's).
- **D-8.** **Provenance summary claim: every technical claim in the program's derived documents
  traces to daily journals; three public repos constitute an auditable paper trail.**
  `measured` for the mechanism's existence — source: capstone (`96db8d6`); repo structure (R,
  T, IRS). *Precision constraint: state as "the mechanism exists and is public," not "every
  claim is perfect" — the appended corrections in D-4 are the honest proof that the mechanism
  catches errors rather than a claim that none occur.*

## E. Program conduct facts

- **E-1.** The program was run entirely on personal time and personal equipment. `recollection`
  as to time; `measured` as to equipment (privately purchased hardware, program record).
  **Mandatory inclusion**, stated plainly (see A-4a).
- **E-2.** Program shape: ~16 weeks, structured in phases, on a 4×RTX 3090 consumer host.
  `measured` — source: capstone (`96db8d6`); `training-plan.md`.
- **E-3.** Division of labor across surfaces: design-, prediction-, and prose-reasoning work
  in the chat interface; command-heavy, mechanistic execution in Claude Code. `measured` as
  practiced **from Week 15 Day 1 onward** — the convention enters the record fully formed as
  a "Session type" journal-header convention pairing each session's genre with its tool.
  Sources: `week-15-day1-interference-design-predictions-journal.md` ("Session type: design /
  prediction (web chat)", with explicit handoff — "the execution session (Claude Code) will
  score against [these predictions]" — and a closing "Next session (Claude Code — execution)"
  section); `week-15-day2-cross-tier-interference-isolated-cpu-handoff-journal.md` ("Session
  type: execution (Claude Code). Command-heavy, mechanistic. No design decisions — those were
  locked in [the Day 1 journal]."); `week-15-day3-delegation-architecture-writeup-journal.md`
  ("Session type: writing/reasoning (web chat). No commands, no probes", carry-forward routing
  the next writing task back to web chat); Week 16 Sessions 1–3 all run under Claude Code —
  `week-16-session1-repo-renames-journal.md`,
  `week-16-session2-journal-consolidation-journal.md`,
  `week-16-session3-capstone-journal.md`. (Session 4 — this one — runs in web chat per the
  same rule; its journal does not yet exist and is not cited.) *Evidential boundary — what the record does NOT contain: the switch itself as an
  event or decision; Weeks 11–14 mention neither tool (Week 14's session journals carry no
  session-type header). Honest article framing: "visible in the record from Week 15 on,"
  never a dated adoption. This is the same shape as the capstone G5 ruling.* First-person
  reflection on *where this task-fit thinking emerged* is `recollection` (carried thread from
  Session 3, deliberately parked for this document; ruled out of the capstone because the
  record supports the through-line but not a dated emergence — the Pulse's first-person voice
  is where that reflection belongs, stated as recollection).

## F. Collaboration mechanics (named-and-illustrated depth)

Decision of record: mechanics are **named** in a sentence or two each, with exactly **two
illustrated** in reader-legible detail. The repos carry full depth; the article links rather
than specifies.

- **F-1.** Illustrated #1: **pickup prompts and pre-session gates** (fresh-context session
  starts from a written prompt carrying scope, contract, and gates; git-state and
  ground-truth checks before any work). Evidential structure: the pickup prompts themselves
  are **uncommitted working artifacts** — the article describes them in first person
  (`recollection`) and may quote a few lines from this session's own pickup as illustration
  (author quoting his own working document, presented as such, not as record). Their
  **committed footprint** is `measured`: the journals' closing carry-forward sections — e.g.,
  `week-15-day1-interference-design-predictions-journal.md`'s closing "Next session (Claude
  Code — execution)" section (already cited in E-3) — which record the handoff each pickup
  was built from. *The article must not imply the prompts are in the repo.* Rationale claims,
  first-person (`recollection` + `interpreted`):
  - **Context isolation:** one subject per session all but eliminates the risk that adjacent
    information enters the context and influences outcomes ("context rot"). The article may
    use the term with a one-clause gloss.
  - **End-of-session prompt-writing as compaction:** at session close, the context is fully
    wired with what the next session needs — which makes that the ideal moment for the AI to
    write the next session's pickup prompt itself; the prompt is, in effect, a deliberate
    context compaction. `recollection` for the practice (the prompts are not in the record);
    the committed carry-forward sections are the checkable trace of the handoff pattern.
  - **The standing big picture:** the sessions run inside a project whose knowledge holds the
    training plan, so per-session focus never means loss of program-level context — narrow
    working context, persistent big picture. `recollection` as operational assessment.
- **F-2.** Illustrated #2: **the inventory-before-drafting rule** (D-6), including that this
  article followed it. `measured` — sources as D-6.
- **F-3.** Named only: explicit IN/OUT scope declarations; per-document review stops with
  inline evidential tags stripped at final; one-experiment-at-a-time as a session discipline;
  the surface-routing rule (E-3). All `measured` as standing practice — collective source:
  session journals and pickup prompts throughout; no individual citations required for
  named-only mentions, but no named element may lack a real basis in the record.
- **F-4.** Character of the collaboration: Claude as a design partner **bound by the same
  contract as the rest of the program** — reviewable, correctable, and constrained by the
  provenance rules; not an oracle. `interpreted` + `recollection` for the characterization;
  the binding mechanisms themselves are D-6/F-1/F-2 (`measured`).
- **F-5.** **Voice and partnership transparency.** The article is written in a deliberate
  "we" for the program's work, defined explicitly and early: the partnership is Mathias
  (author) and Claude (tutor, scribe, lab partner). `measured` for the collaboration's
  existence and mechanics (D-6, F-1–F-3, this inventory's own production); `interpreted` for
  the role characterization. *Constraints: (a) accountability is singular — judgment calls,
  commits, and the published text are the author's; the "we" covers the work, not the
  responsibility; (b) the definition appears before or at first use of "we," so the voice
  reads as transparency rather than vagueness; (c) nothing in the framing may imply Anthropic
  involvement in or endorsement of the program (reinforces G-2's constraint).*

## G. Mentions (no-inflation register)

- **G-1.** NVIDIA / hardware accessibility: six-year-old consumer GPUs (RTX 3090, launched
  2020), bought used, ran the full 16-week inference-engineering program end to end, including
  tensor-parallel serving over NVLink. `measured` — source: program record; capstone
  (`96db8d6`). Stated as an accessibility *fact*, not an endorsement. If rendered as an
  @-mention: em-dashes, never parentheses (standing LinkedIn rule).
- **G-2.** Anthropic / Claude: honest, specific praise — the collaboration structure was an
  enabler the author could not have replaced with solo study; what Claude provided was a
  design partner bindable by the program's own contract (F-4), available at the depth and
  cadence the program demanded. `recollection` (first-person assessment). *Register
  constraint: specific over superlative; no capability claims beyond what the record shows;
  no implication of Anthropic involvement in or endorsement of the program.* Same @-mention
  rule as G-1.

## H. Forward gesture (close)

- **H-1.** The hardware is being retired and upgraded; the author will pursue the next level
  of training following the same method. `stated-intent` + `measured` for the retirement
  (hardware transition is in the record). *Constraints: one or two sentences; no new-hardware
  specifics; no successor-program description; no timeline commitments.* Structural role: the
  thesis demonstrated — the method transfers, the measured values do not.

## I. Exclusions (claims the article may NOT make)

- **I-1.** No career-transition framing, in any phrasing, including quotes that import it.
- **I-2.** No claim that the knowledge-recall → first-principles endpoint has been reached
  (see C-6).
- **I-3.** No employer names.
- **I-4.** No chat-only episodes presented as evidenced (no repo trace → `recollection` at
  most, or excluded).
- **I-5.** No successor-program content beyond H-1 and B-4's artifact-framing note.
- **I-6.** No Markdown tables (LinkedIn Pulse rendering); no tabular content anticipated —
  if any emerges, fenced-code-block ASCII from the start.
- **I-7.** No capstone edits or re-litigation; if drafting surfaces a real capstone error, it
  is logged for Session 5 as an appended-correction discussion.
- **I-8.** Length ceiling: target ~160–200 source lines (~1,800–2,200 words); hard constraint —
  nowhere near capstone length (~500 lines).
- **I-9.** **No measurement values in the article body.** The article states epistemic shapes
  and method conduct, and cites the record; percentages, throughput figures, memory sizes, and
  regime specifics belong to the capstone and journals. A number appears only if a single
  number *is* the point being made (e.g., program-shape facts like "16 weeks" or "three
  repos"), never as technical evidence.

---

## Appended corrections

**Correction 1 — 2026-07-15.** Restructure at author's direction after first review pass;
three changes, no claim retractions:

1. **Thesis emphasis.** The article's primary message is how AI partnership enables
   structured self-training; the method principles become supporting material rather than
   the spine. All existing claim licensing stands; Sections E and F move to the article's
   center, Section D to a compact summary plus pointer.
2. **New deliverable under this same inventory:** `R/docs/training-method.md` — a repo
   document receiving the full method statement and the forged-by evidence (Sections D-1
   through D-8, with their sources). The Pulse links it as it links the compendium. The
   Pulse-specific depth constraints (D-3 depth constraint, I-9 no-measurement-values rule)
   continue to bind the Pulse but do not bind the repo document, which follows normal repo
   conventions: claims trace to journals; technical specifics and values are permitted with
   citation.
3. **Added claims (Pulse):**
   - **E-4.** The training plan (`training-plan.md`) is the program's backbone document:
     phased structure, session sequencing, and a Key Changes section recording amendments
     across the program's life. `measured` — source: `training-plan.md` (incl. §Key
     Changes). The plan's amendment history is citable as partnership-and-method evidence:
     changes were recorded, not silently rewritten (consistent with D-4's discipline).
   - **E-5.** The plan was designed and amended in dialogue between the author and Claude
     across the program. `recollection` + `interpreted` for the co-design characterization
     (the plan document records the amendments, not their authorship); the article states
     this in first person, not as record.

**Correction 2 — 2026-07-16.** Two items surfaced during the article review pass, batched per
the session's standing decision; no retractions:

1. **C-2/C-4 sharpened.** The committed C-2 wording ("the ability to discuss the domain, not
   to debug it") overstated the pre-program position for the model layer specifically. As
   corrected in review, the article asserts: pre-program conversational fluency covered ML
   fundamentals and application-layer topics (per C-4's diagnosis); the model layer was not
   merely undebuggable but largely unknown territory — tensor parallelism, cited in the
   article, is one example of a model-layer concept the author had little or zero awareness
   of before the program, not the sole gap. `recollection`, as before. C-2's domain-level
   claim (and the title resting on it) stands; this narrows its application at the model
   layer.
2. **G-1 extended (new claim, Pulse).** Beyond G-1's committed accessibility facts, the
   article additionally asserts: the six-year-old hardware remained supported by current
   serving frameworks and the current model roster throughout the program, and where the
   hardware's age imposed limitations (capabilities the silicon lacks), readily available
   workarounds proved effective. `recollection + interpreted` for the assessment; the serving
   stack and model roster running on this hardware end to end is program record throughout.
   The article states this without naming specific formats or workarounds (depth consistent
   with the Pulse's register).

*(further corrections, if any, appended below — dated, never by editing the entries above)*
