# Week 15 Summary — Cross-tier interference: predicted, measured, isolated — and the delegation write-up

**Dates:** Day 1 undated in its journal (committed 2026-07-07); Days 2–3 both 2026-07-08
**Substrate:** frozen production stack — 4× RTX 3090, pinned `vllm/vllm-openai:v0.23.0` (`sha256:6d8429e3…22ed8f`); 31B-QAT orchestrator TP=2 on the NVLink pair (:8000), two 12B-QAT workers TP=1 on GPUs 1 and 3 (:8001, :8003), nginx front door (:8080), Prometheus + node-exporter
**Provenance:** all Week 15 results carry tool SHA `T@88493e1`; result commits in R: `eee63a8` (boot), `079fecf` (baselines), `825b3c8` (R2), `8617dbe` (R3)

## TL;DR

Week 15 delivered the operational proof the delegation architecture had owed itself since Week 11: **the two-tier box works as a concurrent system, not three configurations that each boot.** The week ran as a clean three-act predict-before-measure arc — Day 1 locked the experiment design and committed falsifiable predictions before any measurement; Day 2 executed and returned the more informative of the two anticipated shapes (**both tiers isolated**, with a short-context host-coupling signature on the x1 worker that decays to ~0 at depth), plus one honest mechanism correction and, in a same-day appendix, a second-pass re-analysis that tightened the mechanism further and relabeled the operating point; Day 3 finalized `docs/delegation-architecture.md` under the full derived-document contract, closing the arc opened by Week 11's "neither config serves the use case." No verdict moved between prediction and publication; several claims got *narrower*, which is the register working as intended.

## Day 1 — design of record and committed predictions (no measurement)

The design session (web chat) locked, before any execution: the regime matrix (R1 solo baselines re-established on the frozen config — the Week 13 Day 3 numbers were declared **a prior, not a re-confirmed result**, having been taken on the pre-convergence boot path; R2 = 31B victim under worker flood via the nginx pool, which folds in the R4 upstream-split capture; R3 = 12B victim under other-worker + 31B flood), probe sizes [512, 4096, 49152], a **blind isolation threshold** (decode *and* prefill degradation ≤3% at every size), a ≥95% sustained aggressor-saturation gate that voids non-compliant runs, and per-core CPU corroboration from node-exporter. Crucially, the threshold carried a pre-authorized escalation contingency (if any solo baseline shows >1.5% run-to-run spread, the bar becomes 2× the noise half-width) — which Day 2 invoked rather than improvising.

Scope preamble, carried with every number since: this is a **terminal characterization of a retiring topology**. The four GPUs are physically separate, so the only possible interference channel is host-side; a single-die successor measures a *different mechanism*, and trending its values against these is a category error. What transfers is the method and the closed proof, not the numbers.

The committed predictions: both regimes isolated; R3 the largest cell, concentrated at short context; and the **falsifiable commit** — near-ceiling degrades *less* than 512, else the host-handoff-rate mechanism is wrong. Mechanism as committed: host CPU-core contention *and* softirq/interrupt-rate; explicitly not PCIe bandwidth (steady-state serving moves only token IDs) and not NVLink.

## Day 2 — execution: both tiers ISOLATED, every prediction held, one mechanism correction

Sequence discipline held throughout: tool fixes committed clean *before* any results run (worker2 port canonicalized to :8003 across T and R docs — the stale :8002 preset was a silent-404 hazard that would have **faked an isolation result** by flooding a non-listening port; `loaded_window` timestamps added to the probe so per-core CPU could be pulled over exactly the loaded interval), then boot, then empirical placement (UUID→PID join, all `placement_ok`), then baselines, then R2/R3.

Two findings would have voided the run if missed: **`start-stack.sh` ships a stale orchestrator default** (the old FP8 model at MML 33,024 — booting it would have measured the wrong model *and* silently broken the 49,152 probe point; overridden by flags, tool fix recommended and carried forward — since landed with the Week 16 Session 1 rename commit, `T@9801227`), and the nginx pool's :8003 reconciliation was verified live before the flood.

**R1 and the escalation:** decode baselines were clean (run-to-run spread ≤1%), but prefill@512 spread ~4% on both victims — so the pre-authorized contingency fired: prefill bars escalated to ~4%/3%/3%, and **decode became the primary isolation signal**. Applying a committed contingency, not a new decision.

**Results (degradation vs solo baseline; saturation gates PASS sustained in both regimes):**

| Regime | decode @512 / @4096 / @49152 | verdict |
|---|---|---|
| R2 — 31B victim | 0.05% / 0.11% / 0.25% | **isolated** |
| R3 — 12B victim | 0.41% / 0.49% / 0.06% | **isolated** |

The **falsifiable commit HELD** (0.06% at 49K < 0.41% at 512), R3 was the largest cell as predicted, and the nginx split came in 680/663 (50.6/49.4) — the even-split result now confirmed on the v0.23.0 boot path. Full scorecard: all five committed predictions held, with four recorded refinements (the point-predictions that landed inside noise bands are logged as such, not claimed as hits).

**The honest mechanism correction:** softirq stayed at ~0.00% under load, so the interrupt half of the committed mechanism was not observed. Host corroboration showed the real channel — mean logical-CPU busy rising 1% → 41–46% with RAM dead flat and no thread pegged. The shared host resource is *loaded but not exhausted*, which is exactly why the tiers stay isolated. The rate-dependence prediction held; the named resource was half wrong. Recorded as a resolution, not an edit.

**The same-day resolution appendix** (appended per the never-rewrite convention) tightened three things:

- **Hardirq re-analysis:** the interrupt channel would be serviced in `irq`, not softirq — so Day 2's dismissal had measured the wrong counter. The re-pull showed `irq` = 0.000% in all windows, with the register caveat kept deliberately: these counters measure servicing *time*, not interrupt *rate*, so the disposition is **"unsupported and unnecessary"** (host-CPU contention fully accounts for the ≤0.5% coupling), not "proven absent."
- **The coupling is userspace-dominated** (~85% user / ~15% system): the runtime's own per-token Python/tokenize/sampling/SSE work contending on the 6-core host — refining Day 2's "system-time" shorthand.
- **Operating-point relabel:** the empirical 31B-QAT KV ceiling is 193,837 tokens (Day 1's "~54K" figure was wrong by ~3.5×), so **49,152 is a large-context point (~25% of ceiling), not "near-ceiling"**; true near-ceiling isolation (150K+) is **extrapolated, not measured** — safe by the monotone decay. And the decay signature itself sits at ~1× decode noise, so it is *consistent with* the rate mechanism rather than a clean measurement of it.

## Day 3 — the delegation-architecture write-up (writing only)

`docs/delegation-architecture.md` was finalized under the full derived-document contract: 15-claim tagged inventory before prose, review draft with inline register tags, tags stripped at final with `argued` claims converted to one-clause prose traces. The substantive review findings matter beyond the document:

- **The motivating argument's ceiling leg did not survive the QAT migration, and the review caught the draft resting on it.** Week 11's "neither config serves the use case" was measured on the FP8 orchestrator; the QAT migration roughly tripled the TP=2 KV budget, so a single QAT orchestrator serves the validated 131K envelope interactively. The section was restructured constraints-first, with the FP8 numbers standing as the historical record of what forced the decision, not the live argument.
- **The counterfactual is marked as a judgment**, not a derivation: had the program started on QAT, the same decision would likely have followed from the other two constraints.
- **Zone-attribution honesty held:** the even upstream split is credited to `least_conn` on symmetric backends, not to the Week 14 zone fix (a null result); Week 13's missing-zone diagnosis is presented as the hypothesis it was.
- **One claim deliberately withheld as unsupported-by-the-record:** the recollection that the 12B's 131,072 MML pin was later fixed upstream has no committed citation, so the document carries only the temporal hedge.
- The appendix's honesty constraints were verified present in the final (49,152 labeled large-context; deep-context isolation as extrapolation; terminal-topology scoping; "unsupported and unnecessary"; the open 131K–262K worker range).

The Day 3 journal recorded Week 15 as closed with both deliverables done (interference characterization, Days 1–2; architecture write-up, Day 3).

## Not measured this week

- **True near-ceiling isolation (150K+ context)** — extrapolated from monotone decay, stated as such everywhere it appears.
- **Interrupt/DMA-completion *rate*** — the counters used cannot measure it; the hypothesis is disposed of as unnecessary, not refuted.
- **Week 13 Day 3's interference numbers were not re-confirmed** — used only as priors for the predictions.

## Open at week close (as the dailies left them)

- **`start-stack.sh` stale `MODEL_31B`/`MML_31B`/`UTIL_31B` defaults** — the would-have-voided finding, carried as a tool fix at week close. (Resolved shortly after: the fix landed with Week 16 Session 1's rename commit in T.)
- **Optional:** source the upstream fix for the 12B MML pin, if it exists, to strengthen the write-up's hedged sentence.
- **Week 16 — program conclusion** (renames → journal consolidation → capstone → method Pulse), planned as separate sessions.

## Methodology lessons logged this week

- **Commit the escalation rule with the threshold, blind.** The prefill noise that would otherwise have forced a mid-run judgment call was handled by a pre-authorized contingency.
- **A falsifiable commit sharpens a prediction into an instrument** — "near-ceiling degrades less than 512, else the mechanism is wrong" is what made the decay shape evidential.
- **Fix silent-failure hazards before the run:** the :8002 port drift and the stale stack default would each have produced a plausible-looking wrong experiment (a fake isolation verdict; the wrong model at the wrong ceiling).
- **Measure the counter you actually hypothesized:** the softirq dismissal had never measured hardirq; the appendix re-pull closed that gap — and then stated honestly what the right counter still can't show.
- **Append, never rewrite:** the Day 2 body stands; the resolution narrows claims on top of it.
- **Ground truth over preset, every time:** empirical placement, empirical ports, empirical ceilings (Day 1's ~54K → 193,837 correction).

## Artifacts produced this week

- Week 15 results in R (`results/`): boot choreography, four solo baselines (2 runs × 2 victims), R2 + sweep, R3 + sweep, hardirq re-analysis JSON — all carrying `T@88493e1`
- Tool changes in T: worker2 :8003 canonicalization, `loaded_window` block in `interference_probe.py`
- `docs/delegation-architecture.md` — the Weeks 11–15 synthesis, finalized
- Three per-day journals (Day 2 with its appended resolution)
