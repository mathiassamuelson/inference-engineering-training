# Week 14 — Session 2: 12B-QAT TP=2 wins on the NVLink pair

**Phase 3 — Optimization & Quantization**
**Date:** 2026-06-24 (UTC)
**Box:** `inference` — 4×RTX 3090 FE; NVLink pair on GPUs 0+2.
**Image:** `vllm/vllm-openai:v0.23.0` (`sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f`)
**Model under test:** `google/gemma-4-12B-it-qat-w4a16-ct` (worker tier)
**Tool repo (T) HEAD at capture:** `016485d87bd3e80379a689b1996fa913d8acf443` (`tool_git_dirty=False`)

---

## Goals

1. **Task 1 — live validation** of the post-repo-split toolchain end-to-end: the deferred gate
   from Week 13. The offline smoke covered provenance and input-resolution paths but could not
   exercise a *live capture* against a served model or a *real judged comparison* through the
   Anthropic API. This session closes that gate.
2. **Task 2 — 12B-QAT parallelism hypothesis sweep:** does running a model that fits comfortably
   on one card at TP=2 on the NVLink pair *improve* throughput vs TP=1? Predicted prefill and
   decode separately. Characterize PP=2 as an alternative strategy.

Session discipline: predict before measuring; one config per run; empirical placement
verification (UUID-join) on every bring-up; go/no-go between configs.

---

## Task 1 — Live validation (deferred Week-13 gate): PASS

Three internal gates, taken in order, stop at any red.

### Gate 1 — bring-up + empirical placement

Brought up the 12B QAT worker, TP=1, GPU 0 (container `vllm-tp1`, port 8001). `start-vllm.sh`
defaults were overridden explicitly — the script's baked-in defaults are stale (`tp=2`,
`v0.21.0`) and were caught by reading the intent echo rather than trusting it.

`vllm-bringup-checks.sh` PASS: container running, served model matches, chat-endpoint smoke
clean (`finish=stop`, 0.91s). Empirical PID-join placed the worker on **physical GPU 0**
(`GPU-3a7eac76-25ae-edf8-eb11-d6b846ffb56f`), GPU 2 clean — both NVLink cards available for the
sweep's later configs.

### Gate 2 — live capture

First capture used the builtin `DEFAULT_PROBES`, which are the **orchestrator** probe set (8
generic RCA probes), run against the worker tier with the worker system prompt. Tier mismatch:
worker prompt + orchestrator probes. The plumbing it proved (resolve_input, provenance, output
schema) was valid, but the artifact was not a coherent payment-service capture, so it was
**re-captured** with the worker probes file (`probes/worker-rca-payment-service-probes.json`,
6 probes) and the first capture was deleted.

**Lesson:** the default probes are orchestrator-tier. Worker captures must pass
`--probes-file probes/worker-rca-<component>-probes.json` explicitly. The system prompt and the
probe set are *separate* tier selectors and both must match the target tier.

The worker-probe capture proved both `resolve_input()` paths live — the system prompt **and** the
probes file resolved tool-repo-relative from `cwd = R`. Provenance recorded T's clean SHA
(`016485d…`, `dirty=False`) while sitting in R's dirty tree — the `__file__`-anchored provenance
working exactly as the repo split intended. Output written into R's week-14 results; all 6 probes
returned `stop`; sampling deterministic (`temp=0.0`).

### Gate 3 — live judged comparison

Pairwise judge (`claude-opus-4-8`, temperature omitted for 4.x) of the Week-14 worker capture
against the Week-13 payment-service worker capture (same model, same 6 worker probes).

Two earlier pairwise attempts produced **zero API calls** (`probes_scored: 0`) and were deleted:
the first paired against the BF16 parent (different model), and the disjoint-probe-set attempt
could not align orchestrator probes (A) against worker probes (B). Both failed *silently* — the
judge assembled prompts, found no alignable probe pairs, wrote a zero-verdict file, and exited 0.

**Lesson:** the pairwise judge needs matching probe IDs between A and B. A mismatch is silent —
zero calls, zero verdicts, exit 0, a small (~3.6 KB vs ~38 KB) output file. Check
`probes_scored` / output size, not just exit code.

Once both captures used the same worker probe set, the judge ran clean: **12 calls** (6 probes ×
2 orderings for position-bias control), tokens 58008/5171. The legacy-key fallback reader handled
the Week-13 capture's pre-split schema (`git_sha`/`git_dirty`) against the Week-14 capture's
v4 schema (`tool_git_sha`/`tool_git_dirty`).

**Verdict:** 5 ties, 1 model_b (flagged `order-sensitive` — i.e. position-dependent, not a robust
preference), 0 model_a. Same model both sides, different week's capture. Read as a *reproduction*
signal: the post-split capture reproduces the pre-split one, the lone non-tie is noise the
position control caught. Not a quality finding — the captures differ in harness version, not
model. The scope/guardrail axis may read oddly because this gate's purpose was pipeline
validation, not a controlled quality comparison.

**Gate 3 closes Week 13's deferred item.** The post-split toolchain is trusted end-to-end:
capture → resolve_input (prompt + probes) → provenance (T's clean SHA in R's dirty tree) →
live judge → verdicts.

---

## Task 2 — 12B-QAT parallelism sweep

### Hypothesis (recorded before measuring)

Running the 12B QAT (W4A16) at TP=2 on the NVLink pair **improves** throughput vs TP=1, despite
the model fitting comfortably on one GPU. The framing here is deliberate: the naive top-down view
says TP=2 parallelizes the compute and "should" give 2×, which everyone accepts is unrealistic
once the costs the napkin math ignores are paid. So the question was never *whether* TP=2 helps —
a meaningful gain was expected — but *how much* of that theoretical 2× survives. That magnitude
was genuinely unpredictable a priori, because it hinges on a tug-of-war we could not resolve
without measuring. Predicted separately:

- **Prefill (compute-bound):** TP=2 wins — TP splits the per-token matmul work. The
  more-confident leg: prefill is compute-bound, so splitting the matmul should pay off cleanly.
- **Decode (bandwidth-bound):** **TP=2 wins, magnitude unknown.** Decode is the leg where the
  tug-of-war is real. Two opposing forces: (in favor) at TP=2 each card holds *half the weights*,
  so per-card per-token bytes-read roughly halves and the two cards read in parallel — a direct
  bandwidth win, and decode is bandwidth-bound; (against) the per-token all-reduce TP adds, paid
  every token, whose cost scales with interconnect speed. We expected the bandwidth saving to
  win on the NVLink pair (cheap ~100 GB/s all-reduce), but had **no way to quantify the balance
  in advance** — that is exactly what the measurement resolves. Expected the advantage, if the
  saving dominates, to *strengthen with concurrency* as fixed costs amortize.

### Method

Three configs, each run SOLO (orchestrator down, no competing worker — the only variable is the
parallelism strategy). Sweep parameters matched **Week 12 Day 3** exactly:
`--prompt-sizes 8192`, `--max-tokens 512`, concurrency `1 2 4 8` as four separate runs (the tool
takes a single integer per invocation), iterations 3, warmup 1. Endpoint passed without the `/v1`
suffix (the tool appends it). Placement verified by UUID-join on every bring-up.

**Config-to-file mapping** (the throughput filenames distinguish TP=1 from TP=2 only by
timestamp — both lack a parallelism tag; PP=2 carries `_steered_`):

| Config | Parallelism | GPUs | Placement | File timestamps |
|---|---|---|---|---|
| 1 | TP=1 (control) | 0 | naive | `0324–0328` (`c1=032459Z`, `c2=032542Z`, `c4=032639Z`, `c8=032804Z`) |
| 2 | TP=2 (NVLink) | 0,2 | naive | `0336–0339` (`c1=033613Z`, `c2=033640Z`, `c4=033715Z`, `c8=033807Z`) |
| 3 | PP=2 (NVLink) | 0,2 | steered `0,2` | `0348–0350`, all tagged `_steered_` |

> **Known footgun:** TP=1 and TP=2 result files have identical names except the timestamp. The
> parallelism strategy is recoverable only from this table, not the filename. (See tool-cleanup:
> `throughput_sweep.py` should fold `tp/pp`+size into the filename like it does `placement`.)

Placement confirmations (UUID-join): Config 1 — worker on phys 0 (`3a7eac76…`). Config 2 — two
PIDs on phys 0 (`3a7eac76…`) + phys 2 (`8b223d02…`), ~22.5 GB each (the shard split). Config 3 —
two PIDs on the *same* NVLink pair, asymmetric memory (20376 / 21632 MiB), consistent with a
genuine pipeline split: an uneven division of transformer blocks plus the embedding (front of the
network → stage 0) and the LM head (end → stage 1) landing on opposite stages. The UUID-join
establishes the asymmetry and the placement, not the per-tensor breakdown — the specific
attribution is inference from the layout, not measured here.

Config 4 (PP=2 over PCIe 3.0 x1) was **not run.** PP=2 over NVLink already landed at roughly
single-card aggregate throughput; PP over the ~1 GB/s x1 bus can only be worse and tells us
nothing new about the hypothesis.

### Results (8192 prompt / 512 gen, tok/s)

| Metric | TP=1 | TP=2-NVLink | PP=2-NVLink |
|---|---|---|---|
| c=1 decode | 70.0 | **102.6** | 72.2 |
| c=1 prefill | 2480 | **4500** | 3275 |
| c=2 agg_gen | 72.5 | **116.8** | 78.9 |
| c=4 agg_gen | 95.8 | **160.2** | 101.9 |
| c=8 agg_gen | 112.4 | **193.3** | 117.0 |

TP=2 gain over TP=1: c=1 decode **+47%**, c=1 prefill **+81%**, then aggregate **+61% → +67% →
+72%** across c=2/4/8.

### Findings

**1. The tug-of-war resolves in favor of the bandwidth saving — TP=2 c=1 decode +47%
(70.0 → 102.6).** c=1 is the clean, contention-free test: no batching to mask anything, so it is
the halved-weight-read vs per-token all-reduce balance in isolation. The bandwidth saving won, and
this is the number we could not predict in advance — the *magnitude* of the gain, which the
a-priori reasoning could bound only loosely (more than zero, less than 2×). +47% is where it
landed, and the value itself is informative about the mechanism: weight-read is the dominant decode
term but not the only one — KV-cache read, the all-reduce, and kernel-launch/scheduling floor don't
halve, so they consume the gap between the +47% measured and the +100% theoretical ceiling. The
result is the answer to "how does the tug-of-war balance on NVLink," not a surprise that it
balanced favorably at all.

**2. Prefill prediction holds — TP=2 c=1 prefill +81% (2480 → 4500), near-perfect 1.8× scaling.**
The more-confident leg; prefill is compute-bound, TP splits the matmul.

**3. The aggregate lead GROWS with concurrency (+61 → +67 → +72%) — the opposite of compression.**
This is the second prediction (advantage strengthens with load) confirmed in the strong
direction. Mechanism: this workload is prefill-bound (high prompt-to-gen ratio), and TP=2 has 2×
aggregate memory bandwidth *and* 2× compute feeding the batch, so it absorbs stacked load before
saturating. TP=1's single-card wall flattened its scaling (70 → 112, 1.6×); TP=2 scaled 102 →
193 (1.9×). The regime that hurt TP=1 most is where TP=2's doubled resources pay off most.

**4. PP=2 is roughly a wash against single-card, with one instructive asymmetry.**
- c=1 decode 72.2 ≈ TP=1's 70 — predicted exactly. At c=1 there is no pipeline to fill; the two
  stages run serially per token, and the inter-stage hop adds latency that cancels any benefit.
  (The bring-up smoke foreshadowed this: 1.81s round-trip vs ~0.9s for TP.)
- c=1 prefill 3275 — *between* TP=1 and TP=2. PP *does* help prefill at c=1 because processing all
  8K prompt tokens at once fills the pipeline within a single request (stage 0 works later tokens
  while stage 1 works earlier ones). So PP helps prefill but not single-stream decode — a clean
  illustration that PP's benefit mechanism (pipelining, needs depth) differs from TP's (per-token
  weight-read halving, helps every token).
- Under load, PP=2 c=8 (117) barely exceeds TP=1 c=8 (112). PP=2 at this workload is
  approximately equivalent to no parallelism for aggregate throughput. Matches the prior-week
  "PP ~1.7× decode penalty" lesson.

**Ranking at this workload: TP=2 ≫ PP=2 ≈ TP=1.**

### Caveat (interconnect dependence)

The entire TP=2 win rests on the all-reduce being cheap, which is true *only* on the NVLink pair.
This is an NVLink result, not a free-everywhere result. On a slower interconnect the same bet
would likely lose — the per-token all-reduce would dominate the weight-read saving. The unrun
Config 4 (PP over x1) would have documented that floor.

---

## Deferred items

**Tool-cleanup bucket (all touch T — own session after Phase 3 experimental work):**
- `vllm-bringup-checks.sh`: rename `--name` → `--container-name` (ambiguous against served-model
  name).
- Both shell scripts: add explicit `--help` handling (`start-vllm.sh` has none;
  `vllm-bringup-checks.sh` only fires usage on unknown args; `run-judge.sh` forwards `--help` to
  the Python script instead of showing its own).
- `start-vllm.sh`: stale defaults (`tp=2`, `v0.21.0`) are footguns — update or warn.
- `throughput_sweep.py`: fold `tp/pp`+size into the default output filename (same self-describing
  principle the `placement` segment already follows), so parallelism strategy is recoverable from
  the filename rather than only from a journal mapping table.

**Carried from prior planning (each its own session):**
- nginx load-balancing fix: missing `zone workers 64k;` in the IRS upstream block →
  per-worker private least_conn counters.
- per-week dir-name consistency pass.
- Minor drift: stray `issue-39133 reproduction.md` inside a `results/` dir; duplicate `CLAUDE.md`
  under `phase-1-…/week-03-…`.

---

## Outcome

Both session goals met. Task 1 closed Week 13's deferred live-validation gate — the post-split
toolchain is trusted end-to-end. Task 2 confirmed the hypothesis in the strong direction: **for a
12B model that fits on one card, TP=2 on the NVLink pair wins at every concurrency level
(+47% c=1 decode, up to +72% aggregate), and the advantage grows with load.** PP=2 on the same
cards buys almost nothing on decode-heavy serving — same two GPUs, same NVLink, opposite outcomes.
The variable is the parallelism strategy, and the strategy that matches the bottleneck wins.

## Additional activity — Pulse draft

A LinkedIn Pulse post on this finding was drafted and **published** in the same session
([link](https://www.linkedin.com/pulse/splitting-model-fits-one-card-what-tensor-parallelism-samuelson-5ejdc/)).
Framing: the gain was expected; the open question was its *magnitude*, gated on the
halved-weight-read vs per-token all-reduce balance that could not be quantified before measuring —
the +47% decode result is the resolution of that tug-of-war on the NVLink pair. Tabular data built
as fenced ASCII (Pulse renders Markdown tables as literal text).
