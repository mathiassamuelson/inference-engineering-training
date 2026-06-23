# Week 13 Summary — Two-tier QAT quality characterization

## Headline

QAT W4A16 is **quality-equivalent to the BF16 parent at both tiers** of the two-tier RCA copilot
stack — orchestrator (Gemma 4 31B) and workers (Gemma 4 12B) — and is production-ready. The week
also retired all Week-12 serving scaffolding and built the evaluation toolchain that produced the
finding.

## What landed

**Infrastructure**
- Two-tier native stack on a single converged, pinned image (`vllm/vllm-openai:v0.23.0`, digest
  `…6d8429e3…`); all Week-12 scaffolding (source-patched launcher, workaround overrides) retired.
- Confirmed empirically that both 12B variants (QAT + BF16) and the 31B load clean on v0.23.0 with
  **zero** per-model workarounds — the §A retirement holds across the board.

**Quality characterization (the core)**
- **31B orchestrator (Day 8):** QAT ≡ BF16; guardrail adherence an 8/8 tie; decisive verdicts split
  evenly, no order-robust regression on any axis.
- **12B workers (Day 9):** QAT ≡ BF16 on both components (payment-service, order-service); format
  6/6 strict-conformant on both models; pointwise overall 4.83–5.0 — the 12B is **good enough in
  absolute terms** for the focused extraction job, including the hardest outbound-vs-internal
  discrimination.
- **Method:** matched-provenance captures, structured task-appropriate rubrics, position-bias-
  controlled pairwise plus pointwise scoring, and deterministic format checking kept separate from
  the LLM judge.

**Throughput**
- QAT decode outperforms FP8 by +36–50% across the ladder; prefill +1.8–3.9%. (This is QAT-vs-FP8;
  the BF16-vs-QAT story is footprint — ~2.4–2.7× smaller weights on disk.)

**Tooling built and committed**
- `tools/rca_quality_judge.py` — pairwise + pointwise LLM judge, both-orders position-bias control,
  schema-validated output.
- `tools/rca_quality_probe.py` — multi-model capture harness.
- `tools/worker_contract_check.py` — deterministic strict-JSON contract conformance checker.
- `tools/vllm-bringup-checks.sh` — post-launch verification gates (container, placement, log scan,
  endpoint, chat smoke).

**Published**
- LinkedIn Pulse: the LLM-as-judge quality-assessment method (BF16-vs-QAT across both tiers, ~$0.22
  verdict cost), tagging @Google DeepMind and @Anthropic.

## Findings to carry

- **§A retirement holds across all variants on v0.23.0** — no serving workarounds remain.
- **Fenced-example mimicry:** the 12B produces schema-valid JSON but copies a fenced example's
  surface form; fixed by marking examples human-readability-only. Smaller models imitate example
  *shape*, not just content.
- **The 12B's soft axis moves by component** (signal_correctness weakest on payment, extraction_
  fidelity on order) — not uniformly weak anywhere; no axis below ~4.5 mean.
- **Judge robustness:** the `overall`-in-`axes` SchemaError-retry pattern is now standing across Day
  8 and both Day-9 slices; the retry-until-valid machinery earns its place.
- **Provenance:** results and tooling should not share a repo — the Week-14 repo split is the root
  fix, superseding the auto-commit and path-classifier workarounds.

## Artifacts committed

- Prompts: `prompts/worker-rca-payment-service-system-prompt.md`,
  `prompts/worker-rca-order-service-system-prompt.md`
- Probes: `probes/worker-rca-payment-service-probes.json`,
  `probes/worker-rca-order-service-probes.json`
- Rubric: `rubrics/worker-rca-rubric.json` (shared, component-agnostic across both components)
- Journals: Day 8 (31B BF16-vs-QAT), Day 9 (12B worker-component quality, both components)
- Captures, contract-check reports, and judge verdicts for both tiers / both components.
