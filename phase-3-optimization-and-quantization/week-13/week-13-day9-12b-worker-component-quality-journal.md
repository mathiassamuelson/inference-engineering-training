# Week 13 · Day 9 — 12B worker-tier component quality characterization (payment-service + order-service)

## Headline

QAT (W4A16) is **quality-equivalent to the BF16 parent at the worker tier**, and the 12B is
**absolutely good enough** for the focused per-component extraction job — confirmed on both
components characterized. This mirrors the Day-8 finding at the 31B orchestrator tier: QAT W4A16 is
the production choice top to bottom of the stack, now with the worker tier empirically in hand.

The day characterized the *worker* job specifically: focused, single-component evidence extraction
emitting a strict machine-consumable JSON contract — not the orchestrator's open-ended whole-incident
RCA. Two components were taken end-to-end: **payment-service** and **order-service**.

## Result summary

| metric | payment-service | order-service |
|---|---|---|
| format — probes emitting strict-valid contract JSON (of 6) | QAT 6/6 · BF16 6/6 | QAT 6/6 · BF16 6/6 |
| pairwise (QAT vs BF16) | parity (3 order-sensitive) | parity (1 order-sensitive) |
| pointwise overall (mean of per-axis 1–5 scores) | QAT 4.833 · BF16 4.833 | QAT 5.0 · BF16 5.0 |
| pointwise soft axis — lowest-scoring rubric axis (both still ≥ 4.5 mean) | signal_correctness | extraction_fidelity |
| QAT vs BF16 verdict | equivalent | equivalent |

Both models, both components: format at ceiling, pairwise at parity once order-sensitivity is
discounted, pointwise strong in absolute terms. The 12B clears the bar for the focused job.

## Method — and how the worker tier differs from the orchestrator (Day 8)

The worker's job is not the orchestrator's, so the evaluation instruments are purpose-built rather
than lifted from Day 8:

- **Architecture lifted, role authored.** Each worker prompt lifts its component's architecture
  slice verbatim from the operator-copilot system prompt, then wraps it in a freshly authored
  worker role: focused extraction, stay-in-lane scope, no open-ended investigation.
- **Strict machine-consumable output contract.** The worker speaks only to the orchestrator, so its
  output is a strict JSON object — `{component, in_scope, findings[{signal, evidence, confidence}],
  out_of_scope_observations, summary}` — with no prose and no markdown fencing. `evidence` entries
  are verbatim copies from the provided evidence (extraction fidelity made checkable);
  `out_of_scope_observations` is the structural instrument for scope discipline (a cross-component
  signal is routed here, never diagnosed, never fabricated into `findings`); empty `findings` is the
  legitimate "I looked and it's clean" answer.
- **Probes are single-turn extraction tasks, not RCA turns.** The capture harness does one
  request/response per probe with no tool loop, so each probe is self-contained: it hands the worker
  an evidence bundle (logs / metric series / a SQL result, sometimes with a planted foreign signal)
  and asks for structured extraction. This matches the real worker job — the orchestrator gathers,
  the worker distills.
- **Shared component-agnostic rubric.** One worker rubric is reused verbatim across both components,
  because the worker *role* is component-agnostic. Axes: `extraction_fidelity`, `scope_discipline`,
  `signal_correctness`.
- **Format conformance measured deterministically, not by the judge.** The LLM judge is instructed
  to ignore formatting, so asking it whether output is strict valid JSON would be measuring the wrong
  thing with the wrong instrument. Format is therefore a deterministic parser check
  (`worker_contract_check.py`), and the judge scores only the three substance axes. Clean separation:
  the judge does judgment, the parser does parsing.
- **Pairwise and pointwise.** Pairwise (both-orders, position-bias controlled) answers the Day-8
  parity question for the worker tier. Pointwise (absolute 1–5 per axis) answers the question that is
  genuinely open for the smaller model in a way it was not for the 31B: *is the 12B good enough on
  its own terms for the focused job?*

## payment-service

Three artifacts authored (system prompt, 6-probe set, shared rubric) plus the two new tools. Both
12B variants captured under matched provenance (same image digest, same MML, greedy temp-0). Probes
spanned the three §3.3 failure modes (PSP latency cascade, pool exhaustion, credential/auth), the
load-bearing decline-vs-error discrimination, a scope trap (foreign Redis/inventory signals with
payment-service clean), and a nominal-with-benign-decline case.

- **Format:** 6/6 strict-conformant both models (after the fence fix — see Findings).
- **Pairwise:** parity — tie on 5 of 6, a lone weak lean to BF16, and 3 of the decisive-looking
  verdicts flagged order-sensitive (position bias, discounted by the both-orders control).
- **Pointwise:** both 4.833 overall; `scope_discipline` perfect (5.0) on both; the small wobble was
  on `signal_correctness` (QAT 4.667 / BF16 4.5), partly at the decline-vs-error probe — the hardest
  discrimination, doing its job.

## order-service

Same template, validated by the payment-service slice (the fence fix baked in from the start, so the
first capture was already 6/6 — no re-run). Architecture lifted from §3.2: the order lifecycle state
machine, the synchronous inventory + payment calls, pgbouncer transaction pooling, the `orders`
schema. The probe set deliberately included a probe targeting order-service's **outbound-vs-internal
latency discriminator** (is a p99 rise on a synchronous outbound dependency, or internal to
order-service's own pool / GC / query?) — the order-service analogue to payment's decline-vs-error,
and the call most likely to expose a smaller model.

- **Format:** 6/6 strict-conformant both models, first capture.
- **Pairwise:** parity — QAT 2 / BF16 1 / tie 3 with one order-sensitive; discounted, statistical
  noise at n=6.
- **Pointwise:** both **5.0 overall**; `signal_correctness` perfect (5.0) on both — including the
  outbound-vs-internal discriminator probe, which both models handled cleanly (correct outbound
  attribution, payment diagnosis routed out, internal arms correctly resolved to GC / pool). The soft
  axis here was `extraction_fidelity` (QAT 4.667 / BF16 4.5).

A nice cross-check that the discriminator probe was measuring real reasoning rather than a keyword
reflex: the internal-GC probe (outbound flat, GC spiking) was correctly resolved as *internal* by
both models, while the discriminator probe (outbound moving in lockstep) was correctly resolved as
*outbound* — the models read whether the dependency actually moved, not merely whether it was
present.

## Findings

1. **§A retirement holds across both 12B variants.** Both `gemma-4-12B-it` (BF16) and
   `gemma-4-12B-it-qat-w4a16-ct` (QAT) load clean on `vllm/vllm-openai:v0.23.0` with **zero**
   Week-12 workarounds — no `num_soft_tokens` override, no `quantization_config` restatement, no
   `patch_dense` code backport. The QAT load shows `quantization=compressed-tensors` active on the
   Marlin WNA16 kernel with `quantization_config=None` (the checkpoint's native config drives it, no
   shallow-replace trap); the BF16 load shows `quantization=None`, correctly. Since `patch_dense` was
   a unified-architecture bug and the QAT (same architecture) loaded clean, the fix is confirmed
   upstream in v0.23.0 — no per-model serving workarounds remain at the worker tier.

2. **CHECKPOINT: fenced-example mimicry (payment-service slice).** The first payment-service capture
   came back 1/6 (QAT) and 0/6 (BF16) strict-conformant — but `schema_valid` was 6/6 on both. The
   models understood and filled the contract correctly; they wrapped it in a ```` ```json ```` fence,
   mimicking the prompt's own schema example (which was shown fenced for readability). The fix was
   not "say no-fences louder" but to stop modeling the fence: §4 was reworded to demand a raw object
   (begins `{`, ends `}`), the example was explicitly marked human-readability-only, and a hard
   final-line reminder was added. Result: 1/6 → 6/6 on both. The lesson generalizes — a small model
   mimics the *format* of in-prompt examples, so any example must be marked as illustrative, not
   prescriptive of surface form. Baked into the template before order-service, which was then 6/6 on
   first capture.

3. **The 12B's soft axis moves by component.** payment-service's weak axis was `signal_correctness`;
   order-service's was `extraction_fidelity`. The 12B is not uniformly weak on any single axis — the
   friction relocates with the task. No axis fell below ~4.5 mean on either component, which is why
   the absolute (pointwise) read comes out good-enough rather than marginal.

4. **Judge robustness: the `overall`-in-`axes` SchemaError pattern is now standing.** Across Day 8
   and both Day-9 slices, the judge (`claude-opus-4-8`) intermittently returns `overall` folded into
   the `axes` object (and occasionally invents an axis key). The judge's schema validator catches
   each as a `SchemaError` and the retry-until-valid loop recovers cleanly every time. This is a
   reliable wrinkle, not a one-off — the retry machinery has earned its place and should stay.

## Tooling produced (committed this session)

- **`tools/worker_contract_check.py`** — deterministic worker-contract conformance checker. Reads a
  capture, parses each completion strict-first (raw `json.loads`) with a lenient fence-stripping
  fallback that records `needed_fence_strip`, validates against the contract schema, and flags
  `finish_reason == length` truncation separately so a benign cut-off is not misread as a format
  failure. Model identity is read from the capture and propagated to output + filename; optional
  `--expected-component`. Exit non-zero unless every probe is strict-conformant.
- **`tools/vllm-bringup-checks.sh`** — post-launch verification gates: container Running, physical
  GPU placement by PID-join (`docker top` × `nvidia-smi` compute-apps, never trusting `--gpus`
  intent), startup-log scan (hard errors, `patch_dense`, quantization-active, KV-cache lines),
  `/v1/models` reachability + served-id assertion, and a chat-endpoint smoke (Gemma 4 requires
  `/v1/chat/completions`). Used to verify both worker bring-ups this session.

## Decision: split tooling and results into separate repos (parked for a dedicated session)

This session surfaced a repository-structure problem, and the way we arrived at the conclusion is
worth recording because the first two answers were both wrong in an instructive way.

The symptom: capturing a result writes a JSON into a tracked directory, which dirties the tree, so
the *next* capture in a back-to-back pair records `git_dirty: true` even though nothing about the
code or inputs changed. The `git_dirty` flag, meant to answer "was the code+inputs that produced this
result committed?", was firing on a benign sibling result file.

We worked through three answers, each better than the last:

1. **Auto-commit results at the end of a run.** Eliminate the dirt by committing it away. Rejected:
   it makes the measurement tool mutate git as a side effect, and a tool that both writes to disk and
   commits is a latent foot-gun. It also fragments a matched pair across two SHAs.
2. **Classify the dirt by path.** Keep `git_dirty` honest by checking *what* is dirty against an
   allowlist (segment-match on `results`, pattern not week-pinned, in a shared config, default-deny).
   Better — it preserves the strong invariant without mutating git — but it is still machinery built
   to tolerate a situation that should not exist.
3. **Split the repos (the actual fix).** The real issue is that one repo commingles *tools* and
   *data* — the toolchain, and the results/journals the toolchain produces. Separate them: tooling
   into a new repo `T` (`rtx3090-ai-training-tools`), results/journals/captures staying in the
   current repo as `R`. Both repos public. Then a capture writes into R, the SHA that pins
   what-produced-it is T's, and writing results never touches T. T stays
   clean across an entire capture batch with zero classification logic; any dirt in T is once again
   unambiguously a real code change. The dirty-by-sibling problem does not get classified — it
   **disappears**.

The tell that (3) is the right level: it *removes* machinery rather than adding it, and it supersedes
both earlier ideas (auto-commit and path-classification are both dropped). The one technical crux for
the dedicated session: the capture / judge / check tools must record **T's** commit, not the cwd's —
post-split the working directory is R, so a naive `git_provenance(cwd)` would record R's SHA and
relocate the problem rather than solve it. Secondary upside: a public, self-contained toolchain
(position-bias-controlled judge, contract checker, bring-up gates) becomes a citable artifact for the
architecture write-up and the LLM-as-judge article.

Parked as a dedicated session (Week 14+), not a mid-week insert — it touches the harness and deserves
a careful pass.

## State at close

- Both components fully characterized, captured, checked, judged; all artifacts, captures, check
  reports, and judge verdicts committed.
- Worker containers (`gemma4-12b-native-test` QAT, `gemma4-12b-bf16-native-test` BF16) torn down;
  cards returned to idle.
- Five reusable artifacts now in the repo: two worker prompts, two probe sets, one shared rubric,
  plus the two tools above.

## Next

- **Repo split** — dedicated session (above).
- **LinkedIn Pulse, LLM-as-judge / quality-assessment article** — now has both tiers in hand
  (orchestrator Day 8 + worker Day 9): the position-bias-controlled pairwise judge + structured
  rubric + matched-provenance captures as method, the lossless-QAT finding as payload. Tables as
  ASCII inside fenced code blocks (Pulse does not render Markdown tables).
- **Architecture write-up** — unblocked; hold the load-balance claim until the IRS nginx
  `zone workers 64k;` fix lands and a re-probe shows balanced distribution.
- **Remaining components** beyond payment-service + order-service — mechanical repetition now that the
  template is validated; diminishing returns.
