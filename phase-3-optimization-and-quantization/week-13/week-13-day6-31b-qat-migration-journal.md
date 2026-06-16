# Week 13 — Day 6 Journal: 31B FP8 → QAT w4a16-ct migration (§B)

**Session date:** 2026-06-15 evening (US Mountain) / 2026-06-16 UTC
**Phase:** 3 — Optimization and Quantization
**Scope:** §B arc — migrate the orchestrator slot from FP8 to the 4-bit QAT variant of the 31B,
then characterize what the migration buys. Three sub-steps completed: (1) load-fit, (2) context
headroom, (3) quality. The fourth (benchmark) was deferred to keep one-experiment-at-a-time
discipline and to save power/heat.

---

## 1. What we set out to do

Replace the 31B orchestrator's weight format. The incumbent is `RedHatAI/gemma-4-31B-it-FP8-block`
(8-bit). The candidate is `google/gemma-4-31B-it-qat-w4a16-ct` (4-bit, quantization-aware-trained) —
the exact same compressed-tensors format the two 12B sub-agent workers already run. The hoped-for
payoff was three wins: more usable context ("headroom"), faster token generation, and a tier where
both the orchestrator and the workers use the same 4-bit format ("quant consistency").

Both 12B workers (GPUs 1 and 3) stayed up and untouched throughout. Only the orchestrator slot on
the NVLink GPU pair (GPUs 0+2) was mutated.

---

## 2. Starting state (verified, not assumed)

Full native stack was already up from Day 5 (44–47 min uptime). Repos synced and clean; the Day-5
push had landed (HEAD `6887816`, with the §A snapshot and Step-4 rewire in history). Pinned image
digest confirmed (`vllm/vllm-openai:v0.23.0` @ `…6d8429e3…22ed8f`). QAT 31B already cached.

Placement was verified by the GPU-UUID → host-PID → container join (not launcher intent):
`gemma4-31b-tp2` owned both GPU 0 and GPU 2; the two 12B workers were the only processes on GPUs 1
and 3. So the teardown target was unambiguous and the workers were provably safe.

---

## 3. §B-(1): Load-fit on the NVLink pair (TP=2, GPUs 0+2)

Booted the QAT 31B at the **FP8-matched config** (max context 33,024, GPU-memory-utilization 0.95)
so the weight footprint would be directly comparable to the FP8 anchor.

**Result — GO.** Three gates, all green:

- **Kernel:** both ranks selected `MarlinLinearKernel for CompressedTensorsWNA16` — genuine 4-bit,
  no silent fallback to a higher-precision path.
- **Weights footprint:** **9.59 GiB per GPU.**
- **Functional probe:** chat endpoint returned coherent text (correct EOS), and a fresh-PID
  placement re-verify confirmed the new process landed on GPUs 0+2 with the workers' PIDs unchanged.

**Prediction vs measured — a miss worth keeping.** I predicted ~8 GiB/GPU (treating 4-bit as
roughly halving FP8's 15.85). Actual was 9.59 — a realized ratio of **0.605×**, not 0.50×. The
reason is structural and reusable: the model has two **uncompressed bf16 tensors** — the token
embedding and the output head, each over a 262K-entry vocabulary, ~2 GB apiece — that are 16-bit in
*both* the FP8 and the 4-bit checkpoints. Quantization only shrinks the transformer body; those two
big tensors are a fixed floor that 4-bit can't touch. This is the same fact that caused the Week-12
12B load OOM, now showing up as a footprint floor rather than a wall.

Net: the 4-bit orchestrator carries **~6.3 GiB/GPU less weight** than FP8 — real, just less than the
naive halving would suggest.

---

## 4. §B-(2): Context-headroom ceiling walk

Goal: find the highest max-context (`max-model-len`, "MML") at which the KV cache can still hold one
full-length request — i.e. concurrency falls to ~1.0×. The KV cache is the per-request working
memory for attention; a bigger one means more context and/or more simultaneous requests. Method: one
separate boot per rung, MML the only variable, utilization pinned at 0.95, CUDA-graph memory tax
left at default (a separate lever, deliberately not mixed in). The measurement is vLLM's own
boot-time profiler line, plus a chat probe to confirm it serves.

| max context (MML) | KV pool (tokens) | max concurrency | notes |
|---:|---:|---:|---|
| 33,024 | 105,609 | 3.20× | the §B-1 anchor |
| 65,536 | 151,308 | 2.31× | served, coherent |
| 131,072 | 193,837 | 1.48× | served, coherent |
| 262,144 | refused at load | — | needs 11.96 GiB KV > 10.3 available |

**The mechanism — and a corrected mental model.** I predicted the pool would *shrink* as MML rose
(bigger working set). It did the opposite: it **grew** at every rung. The key invariant the logs
revealed is that the **KV memory budget is essentially constant (~10.3 GiB) across all four boots** —
it never moved. What changed was *tokens-per-GiB*: vLLM sizes its paged-KV blocks as a function of
max context, and larger MML yields larger blocks with less per-token accounting overhead, so the
same fixed GiB budget reports *more* tokens. Not more memory — finer accounting. My "pool grows then
plateaus against the memory budget" follow-up guess was also wrong, for the same reason: the budget
was never the moving part.

**The ceiling.** The 262,144 boot refused cleanly at load — and the error message did the arithmetic
for us: a full 262K request needs 11.96 GiB of KV but only 10.3 GiB is available, so vLLM reports an
**estimated maximum model length of 218,624 tokens**. That is the ~1.0× ceiling, handed to us
directly rather than searched for.

This was a **KV-headroom wall, not an architecture wall** — vLLM accepted 262,144 as a valid config
value (it is the model's real max context) and refused only because the KV pool couldn't cover one
request. The suggested fix (raise utilization) was off-limits here: utilization is the pinned
variable.

**Win #1 — context headroom — CONFIRMED.** The 4-bit orchestrator reaches **~218K usable context**
versus FP8's Week-11 ceiling of **54,496** — roughly **4× the headroom**, with no change to the
parallelism strategy or the GPU pair. The deployment target's sessions sit trivially inside this; the
orchestrator is no longer context-bound, and the binding constraint shifts entirely to
concurrency/throughput.

*Caveat for honesty:* 218,624 is vLLM's load-time estimate, not a value we booted-and-served at. If
we ever need it nailed, boot at a block-aligned value just under it and confirm concurrency ≥ 1.0×.
For the architecture write-up, "~218K, ~4× FP8" is the supported and sufficient claim.

---

## 5. §B-(3): Quality validation vs FP8

### 5.1 Deployment-target pivot
The practical deployment target was generalized this session from the old statmon-ai monitoring
assistant to a broader **operator copilot**: an RCA (root-cause-analysis) assistant that reasons over
an architecture plus per-component knowledge, then investigates incidents by reading logs, running
read-only commands on live service components, and issuing SQL queries. This is a more discriminating
quality surface than a monitoring assistant — multi-step reasoning, tool selection, and interpreting
structured evidence are exactly where 4-bit regressions would surface if they exist.

### 5.2 Harness and prompt
Built a multi-model quality harness and a representative ~6K-token operator-copilot system prompt
(a fictional "Meridian" order/payment platform with a full component map, tool surface, data model,
incident playbook, and guardrails). The harness takes the model name as an argument (never
hardcoded), propagates that identity into the request, the console header, the JSON metadata, and the
default output filename (so two models' runs can't overwrite each other), and records the system
prompt's SHA-256 as proof that both models saw an identical prefix.

Eight probes spanning the failure-prone surface: hypothesis formation, log interpretation, SQL
authoring, p99-latency architectural reasoning, read-only command selection, metrics interpretation,
a **mutating-action guardrail** ("just restart payment-service"), and root-cause synthesis.

### 5.3 Two QAT captures (the prompt got fixed mid-step)
- **Capture 1** (prompt ~3,240 tokens) was strong overall but exposed two blemishes: probe 1
  **fabricated a date** for a tool-call time window (the prompt gave a clock time with no date and no
  guidance), and probe 3's SQL used a **wrong `width_bucket` signature** producing subtly mislabeled
  buckets — confident, runnable, and wrong.
- We then (a) padded the prompt to ~5.1K tokens with genuine content (playbook, worked example,
  deeper component signatures, metrics catalog) to exercise the long-prefix behavior the architecture
  targets, and (b) added an explicit **time-handling rule** mandating relative anchors and forbidding
  fabricated dates.
- **Capture 2** (prompt SHA `ba18e9c0…`, ~5.1–5.3K tokens/probe) validated both fixes: probe 1 used a
  relative anchor (`-30m`, no date), and probe 3 produced the **correct 4-argument
  `width_bucket(…/60, 0, 60, 12)*5`**. Notably the larger, more explicit prompt *improved* the weak
  cell rather than straining the model. This became the locked QAT baseline.

### 5.4 FP8 swap and matched comparison
Tore down QAT, booted FP8 at its baseline config (MML 33,024, util 0.95 — the same config as the
committed Day-4 throughput anchor). Footprint and KV reproduced the anchors exactly:

| | weights/GPU | KV budget | KV pool @ MML 33,024 | concurrency |
|---|---:|---:|---:|---:|
| QAT w4a16-ct | 9.59 GiB | 10.3 GiB | 105,609 tok | 3.20× |
| FP8-block | 15.85 GiB | 4.04 GiB | 41,427 tok | 1.25× |

That side-by-side at one identical config is the cleanest single illustration of the headroom win:
same util, same MML, same GPUs — FP8's heavier weights leave it **less than half** the KV pool.

The FP8 boot log also surfaced the benchmark hypothesis's mechanism directly: the GPUs have **no
native FP8 compute** (Ampere), so FP8 here runs **weight-only via the Marlin kernel** — emulated, with
a warning that it "may degrade performance for compute-heavy workloads." QAT's w4a16 is also Marlin
but moves half the weight bytes per token. So the §B-(4) "smaller AND faster" hypothesis now has its
mechanism on record before we even run the sweep.

Ran the identical 8 probes against FP8 (same prompt SHA `ba18e9c0…` — comparison validity gate
passed).

### 5.5 Result: near-lossless
Cell by cell, the two models are **near-indistinguishable** at temp 0:
- Same diagnoses on every probe (PSP cascade, single-partition Kafka stall, internal pool exhaustion).
- **Both used relative time anchors on probe 1** — proving the date fix is prompt-driven, not
  model-specific.
- **Identical corrected SQL on probe 3** (the only difference a cosmetic `ASC`).
- **Both passed the guardrail** (probe 7): refused the blind restart, named the in-flight-charge /
  double-charge mechanism, required explicit confirmation. Instruction-following under pressure is
  intact on 4-bit — the single most reassuring result.
- No cell where FP8 was meaningfully better.

**Win #3 — tier-wide quant consistency — CONFIRMED.** The orchestrator can run 4-bit w4a16 with no
quality regression versus the FP8 incumbent on the operator-copilot RCA surface. Both tiers can now be
the same format.

---

## 6. Three-wins scoreboard

| Win | Status | Evidence |
|---|---|---|
| #1 Context headroom | **Confirmed** | ~218K vs 54,496 (~4×), §B-2 |
| #2 Decode/prefill speed | **Open** | §B-4 benchmark, deferred |
| #3 Quant consistency | **Confirmed** | near-lossless QAT-vs-FP8, §B-3 |

The architecture write-up is two-thirds unblocked. Holding it until the benchmark lands and (separately)
until the IRS nginx `zone` fix lets us re-probe load balancing.

---

## 7. Prediction scorecard (kept honest)

| Prediction | Outcome |
|---|---|
| QAT weights ~8 GiB/GPU | **Miss** — 9.59; bf16 embed+head don't quantize (0.605×, not 0.50×) |
| KV pool shrinks as MML rises | **Miss (wrong direction)** — it grew at every rung |
| Pool plateaus ~152K against budget | **Miss** — kept climbing to 193,837; budget was constant, accounting changed |
| QAT quality near-lossless | **Hit** — confirmed cell-by-cell vs FP8 |
| "Longer reasoning chains degrade first" under 4-bit | **Did not materialize** — p4/p8 were among the best cells |
| FP8 weights 15.85 GiB/GPU | **Hit** — exact |
| Date fix is prompt-driven, generalizes to FP8 | **Hit** — FP8 also avoided fabrication |

The footprint and pool-direction misses share a root: I kept modelling the *memory budget* as the
moving variable when it is the constant. The corrected model (block sizing changes tokens-per-GiB at
a fixed GiB budget) is a post-hoc correction, not a prediction — recorded so the next ceiling walk
starts from it.

---

## 8. Key learnings reinforced / added

- **KV pool token-count rises with MML at fixed utilization** because paged-KV block sizing improves
  tokens-per-GiB accounting — the GiB budget is constant, not the token count. The 1.0× ceiling is
  therefore far above any constant-pool extrapolation, and vLLM's load-time "estimated maximum model
  length" gives it directly. (Reinforces: never extrapolate the pool linearly across MML rungs.)
- **Uncompressed bf16 embedding + output head are a quantization floor** — they're 16-bit in both FP8
  and 4-bit checkpoints, so 4-bit shrinks the body only. Explains the 0.605× (not 0.50×) footprint.
- **A load-time refusal with an "estimated maximum model length" is a gift** — it's the ceiling
  computed for you; no search rung needed.
- **`--gpu-memory-utilization` cannot fix a load-time KV shortfall** — it sizes the pool *after*
  weights load (re-confirmed by the 262K refusal).
- **Ampere has no native FP8** — FP8 runs weight-only via Marlin (emulated). Sets up the benchmark's
  "smaller AND faster" hypothesis with a mechanism, not just a hope.
- **A larger, more explicit prompt can *improve* a weak quality cell** — the SQL bug resolved on both
  models once §5 spelled out bucket-labeling expectations.
- **Prompt-design gaps masquerade as model errors** — the fabricated date was the prompt's fault (no
  date, no anchor guidance), not a quantization signal; fixing it at the prompt level removed the
  confounder and generalized across both models.
- **Startup line necessary but not sufficient; verify placement empirically on every fresh process** —
  held throughout (UUID→PID→container join after each boot).

---

## 9. Artifacts produced

Committed to `~/work/rtx3090-ai-training` (clean tree, SHAs recorded in each result JSON):

- `prompts/operator-copilot-rca-system-prompt.md` — system prompt, SHA-256 `ba18e9c0…`.
- `tools/rca_quality_probe.py` — multi-model quality harness (model identity captured everywhere).
- `phase-3-optimization-and-quantization/week-13/results/exp_quality_rca_gemma-4-31B-it-qat-w4a16-ct_20260616T015747Z.json`
  — QAT quality baseline (git_sha `1a1e3cf2…`, prompt SHA `ba18e9c0…`).
- `phase-3-optimization-and-quantization/week-13/results/exp_quality_rca_RedHatAI--gemma-4-31B-it-FP8-block_20260616T020802Z.json`
  — FP8 matched comparison (git_sha `036afb18…`, prompt SHA `ba18e9c0…`).
- The initial 3.2K-prompt QAT capture (`…20260616T014324Z.json`) was banked as a superseded record.

Result files to commit from this session if not already: the FP8 quality JSON, plus this journal.

---

## 10. Machine state at close

QAT orchestrator was torn down for the FP8 swap; FP8 was booted, used for the comparison, and is
being **torn down at session end to save power and heat**. The two 12B workers (GPUs 1, 3) remained
up and untouched the whole session. At next session start, **establish ground truth** (`docker ps`,
`nvidia-smi`) — do not assume the orchestrator slot state.

---

## 11. Next: §B-(4) benchmark (deferred)

The only remaining §B step: decode/prefill throughput, QAT vs FP8, at matched context, same sweep
ladder, comparison cells pulled from the committed Day-4 FP8 anchor
(`throughput_sweep_…FP8-block_c1_20260614T194625Z.json`). Hypothesis: native-ish 4-bit w4a16 is both
smaller AND faster than Marlin-emulated FP8, decode especially (bandwidth-bound; fewer weight bytes
per token). Sequencing note in the Day-7 pickup. Confirming this closes win #2 and fully unblocks the
architecture write-up.
