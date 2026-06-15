# Week 13 — Day 4: vLLM v0.23.0 convergence + 31B FP8 re-baseline

**Date:** 2026-06-14
**Phase:** 3 (Optimization & Quantization)
**One-line summary:** A single stable vLLM release (v0.23.0) was confirmed to serve *both*
model paths we run, and the large model's performance on it was confirmed unchanged from the
prior version — clearing the way to change the large model's quantization next session.

---

## Background a future reader needs

We run two model "tiers" on one 4×RTX 3090 workstation:

- **The orchestrator:** `RedHatAI/gemma-4-31B-it-FP8-block`, a 31-billion-parameter model in
  **FP8** (8-bit floating-point weights). It runs across two NVLink-bridged GPUs (0 and 2)
  using **tensor parallelism, TP=2** — the model's layers are split head-wise across the two
  cards, which talk over the fast NVLink bridge.
- **The workers:** `google/gemma-4-12B-it-qat-w4a16-ct`, a 12-billion-parameter model in
  **w4a16** (4-bit weights, 16-bit activations), produced by **QAT** (quantization-aware
  training). Each worker runs on a single GPU (TP=1).

Until today these tiers ran on **two different container images**, and neither was a clean
stable release:

- The 31B ran on a **pinned `vllm/vllm-openai:v0.21.0`** image.
- The 12B ran on a **nightly/preview `gemma4-unified`** image, and even then needed a **source
  patch** plus an **`--hf-overrides` blob** to load, because the 12B's architecture
  (`Gemma4UnifiedForConditionalGeneration`, an "encoder-free unified" variant) was not yet in
  any stable vLLM and its 4-bit checkpoint config wasn't parsed cleanly.

This split is fragile: two images to track, a patch to maintain, and no single version that
serves both. **Day 4's job was to find out whether vLLM v0.23.0 ends that split** — and to do
it without changing anything else about the 31B, so any difference we saw could only be the
version.

A recurring term below: the **KV cache** (or "KV pool") is the working memory that holds the
attention state for in-flight requests. It is whatever GPU memory is left *after* the model
weights, CUDA-graph capture, and activations are accounted for — so it shrinks when any of
those grow. **Prefill** is the one-time work of reading the prompt; **decode** is generating
the answer one token at a time. We measure both as tokens/second.

---

## Goal for the day

1. Pull vLLM **v0.23.0**, pin its image digest, and run a **convergence go/no-go**: does one
   stable version load both the 12B unified path *and* the 31B FP8 path?
2. **Re-baseline the 31B on v0.23.0 with the model held constant (still FP8)** and compare
   against the Week 11 measurements, to confirm the version upgrade does not regress
   performance *before* any quantization change.
3. Decide what to do about the CUDA-graph "tax" flag.
4. Land the queued tool fixes.

Discipline held throughout: **one experiment at a time**, **commit before any measured run**,
**verify GPU placement empirically** (never trust the launcher's intent), and **a functional
chat probe is required** — a clean startup log alone is not enough to call a config working.

---

## Pre-session state

Both repos clean and pushed. The pinned image digests for v0.21.0, the gemma4-unified image,
and nginx were all still present (they are our regression anchors — losing them would mean
losing the ability to reproduce the old baseline). Models cached.

**Deviation from the pickup's expectation:** the pickup assumed the stack would be torn down
cold for overnight heat. It was actually still **up** on the old images. This changed nothing
except the sequencing — it meant we *tore down and relaunched* on the new image rather than
booting from cold. The old config is fully reproducible from its pinned digest, so nothing was
lost by stopping it.

---

## Step 2 — Convergence go/no-go → **GO**

**v0.23.0 exists and carries the fix.** The release shipped two days before this session
(2026-06-12), 408 commits from 200 contributors. Its highlights explicitly list
**"encoder-free Gemma 4 Unified support (#44429)"** — the exact thing the 12B workers needed
natively. The host driver supports the CUDA 13 runtime, so the plain `v0.23.0` tag was correct
(no `-cu129` variant needed). Pinned digest:

```
vllm/vllm-openai@sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f
```

**The 31B FP8 path loaded clean on v0.23.0, and its memory footprint was identical to Week 11.**
Every boot-time number matched the prior version at the same settings (context budget 33,024,
GPU memory utilization 0.95, text-only):

| Footprint (per GPU)        | Week 11 (v0.21.0) | Day 4 (v0.23.0) |
|----------------------------|-------------------|-----------------|
| Model weights              | 15.85 GiB         | 15.85 GiB       |
| Available KV memory        | 4.04 GiB          | 4.04 GiB        |
| CUDA-graph capture         | 0.88 GiB          | 0.88 GiB        |
| KV pool (total, both GPUs) | ~41,300 tokens    | 41,427 tokens   |
| Max concurrency @ 33,024   | 1.25×             | 1.25×           |

Placement was verified empirically (the two compute processes sat on the UUIDs that map to
physical GPUs 0 and 2 — the NVLink pair; GPUs 1 and 3 stayed idle), and a chat probe returned
a real answer on `system_fingerprint: vllm-0.23.0-tp2`.

**The 12B path loaded clean on v0.23.0 — natively, with the patch *and* the override removed.**
This is the best-case outcome. We deliberately launched the 12B using the plain launcher
(`start-vllm.sh`) with **no patch mount and no override blob**, and it served. The startup log
confirmed *why* it worked rather than silently degrading:

- `Resolved architecture: Gemma4UnifiedForConditionalGeneration` — the encoder-free
  architecture is now recognized natively (this is #44429 working). **The source patch
  retires.**
- `Using MarlinLinearKernel for CompressedTensorsWNA16`, with `quantization_config=None` — the
  4-bit checkpoint was parsed straight from its own config and the Marlin INT4 kernel selected,
  with no override injected. (**Marlin** is the kernel that runs 4-bit integer weights on these
  Ampere GPUs.) **The override blob retires too.**

12B native footprint banked (single GPU, context budget 131,072, utilization 0.90): **weights
8.28 GiB**, **KV pool 370,575 tokens**, **max concurrency 2.83×**. The 8.28 GiB single-GPU
weight figure is a useful anchor for next session's expectation that the *31B* in 4-bit should
land at roughly half its FP8 footprint.

**Verdict:** one stable image (v0.23.0) serves both tiers. Convergence confirmed.

---

## Step 3 — 31B FP8 re-baseline → **GREEN (non-regression)**

With the 12B test worker stopped (Week 11 measured the 31B solo, so we matched that), the 31B
was relaunched alone on v0.23.0 and re-verified (placement join + chat probe on the fresh
process — a new process gets its own verification, never inheriting the prior boot's). Then the
matched-size throughput sweep was run: concurrency 1, 3 measured iterations per size, 1 warmup,
sizes 512 → 32,768. 18/18 requests succeeded.

**Predict-before-measure (stated before the run):** decode flat within run-to-run noise
(~±1–2%); prefill flat-to-slightly-up *if* v0.23.0's auto-selected kernels happened to be
faster. Null hypothesis "no material change" was the gate-opening outcome.

Results (per-size average, v0.23.0):

| Prompt size | Decode (tok/s) | Prefill (tok/s) |
|-------------|----------------|------------------|
| 512         | 44.3           | 1952.7           |
| 2,048       | 43.5           | 1872.4           |
| 4,096       | 42.3           | 1796.2           |
| 8,192       | 40.2           | 1690.0           |
| 16,384      | 38.1           | 1524.1           |
| 32,768      | 34.8           | 1291.8           |

Against the Week 11 anchors I can cite confidently (512, 2,048, 32,768), every cell lands
within ≤0.25% — and the *shapes* of the curves match: decode eased −21.4% across the ladder
(Week 11: −21.3%), prefill declined −33.8% monotonically (Week 11: ≈−34%). Combined with the
identical boot footprint, **v0.23.0 reproduces the v0.21.0 FP8 baseline end-to-end.** This is
the result that unblocks the next session's quantization work: the version variable is isolated
and non-regressing, so a quantization change will be measured against a clean anchor.

> **Caveat for whoever builds the full cell-by-cell delta table later:** the Week 11 values for
> 4,096 / 8,192 / 16,384 must be pulled from the **committed Week 11 results JSON**, not from
> recollection. (The "31B is faster" surprise on Day 3 came from comparing against a remembered
> number that was actually a different operating point. Same trap; avoid it.)

**Prediction honesty:**

- *Decode flat* — **confirmed**, essentially 0% change.
- *Prefill up* — **did not happen.** Prefill was flat, not up. Likely reason: on these Ampere
  GPUs the FP8 path is emulated via Marlin and prefill is bound by attention cost on the global
  layers plus memory bandwidth, not by the matrix-multiply kernels the autotuner optimizes — so
  there was no faster path to pick. The directional sub-bet was wrong; the load-bearing bet
  (non-regression) was right.

**Warm-up handling — discard turned out unnecessary, verified two ways.** We planned to discard
the first measured size's prefill (a known GPU clock-ramp artifact) and any size that hit a
just-in-time kernel compile mid-run. Neither fired: the 512 prefill read a clean 1952.7 across
three tight iterations (not depressed), and the new `jit_monitor` raised no warnings during the
sweep. The full ladder stands as measured; nothing re-run.

**Committed:** `throughput_sweep_vllm-openai_gemma-4-31B-it-FP8-block_c1_20260614T194625Z.json`
under `phase-3-optimization-and-quantization/week-13/results/`, recorded against clean git SHA
`e4f0a627` (verified the embedded SHA equals HEAD — the code was committed-clean at measurement
time).

---

## A prediction that was wrong, and why it mattered

Going in, I'd flagged that v0.23.0 makes a newer model runner ("Model Runner V2") the default
for some model families, and *if* the 31B silently landed on it, its different CUDA-graph
behavior might make the whole CUDA-graph-tax question moot.

It did **not**. The 31B stayed on the standard runner (`gpu_model_runner.py`), exactly as the
release notes implied (the new runner's default list covers Llama / Mistral / Qwen3, **not**
Gemma-4 Dense). Two consequences:

1. The re-baseline is a clean *version-only* comparison, not a runner-generation confound.
2. The CUDA-graph tax **persists unchanged** — which kept Step 5 a live decision, not a moot
   one.

---

## Step 5 — CUDA-graph-tax flag → decision made, no code change

Background: since v0.21.0, vLLM reserves extra GPU memory up front to profile CUDA-graph
capture. This shrinks the KV pool — the "tax." On v0.23.0 the 31B logged the *identical* tax:
at utilization 0.95 the effective utilization is ~0.9093 (the engine even suggests raising to
0.9907 to recover, or setting `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` to disable).

**Decision:** leave the `--profiler-cudagraphs` flag exactly as it is — an **opt-in** lever in
`start-vllm.sh`. Do **not** promote it to the baseline or wire it into `start-stack.sh`.
Reasoning: at 0.95 the tax is already absorbed with 1.25× concurrency headroom that serves the
workload, and the engine's own recovery suggestion (profiler-off) is the path that ran out of
memory on these 24 GiB cards back in Week 11 — the real ceiling-recovery lever is laddering
utilization up to ~0.97, not turning the profiler off. **No code change.**

---

## Tool fixes committed today

All in the training repo, each its own commit except the last (bundled by request):

1. **`start-stack.sh` — week parameter.** The results path had `week-13` hardcoded; it would
   silently write to the stale week's directory next week. Now a `--week` flag (default
   `week-13`, so current behavior is unchanged) builds the path; `--results-dir` still
   overrides outright.
2. **`start-stack.sh` — clobber guard.** The default boot-result filename had no timestamp, so
   re-booting the same model pair silently overwrote the prior (possibly committed) result —
   exactly what bit us on Day 3. The default filename now carries a UTC timestamp (colon-free,
   matching the sweep tool's convention); an explicit `--out` still bypasses it.
3. **`start-stack.sh` — dirty-tree false-positive.** The "working tree is DIRTY" provenance
   warning fired on uncommitted *result* files, which are expected at write time. The check now
   excludes the results directory, so it warns only on real code/tool changes. Verified with a
   four-case test (clean → quiet, results-only → quiet, tool-modified → warns, reverted →
   quiet).
4. **`throughput_sweep.py` — anchor to repo root, not CWD (one bundled commit, three fixes).**
   The tool trusted the current directory in three places: the results directory was
   CWD-relative (which is why this morning's JSON first landed in the wrong folder); the git
   SHA/dirty status were read from the CWD's repo (a silent provenance-corruption risk if ever
   run from outside the repo or a sibling repo); and the dirty check included the results dir.
   All three now anchor to the script's own location (`Path(__file__)` → repo root), the git
   metadata uses `git -C <repo-root>`, and the dirty check excludes the results dir. Verified by
   importing the edited module from a throwaway repo and exercising clean / results-only /
   tool-modified states from a *different* working directory.

---

## Step 4 — patch/launcher retirement: **demonstrated, execution queued**

The *evidence* for Step 4 is already in hand: the 12B booted clean natively on v0.23.0 with no
patch and no override, which proves the patch file, the override blob, and the
`start-12b-qat.sh` launcher (whose only jobs were to mount the patch and inject the override)
are all unnecessary. What remains is *execution*, deferred to the next block because it deserves
a focused full-stack boot rather than an end-of-day rush:

- Delete the patch file and `start-12b-qat.sh`.
- Rewire however `start-stack.sh` launches the workers to use the native `start-vllm.sh` path.
- **Boot the full multi-tier stack on v0.23.0** (two workers + 31B + nginx) — *not yet done*;
  today validated each tier independently, but the production layout booting native-on-v0.23.0
  is its own go/no-go and is the snapshot the architecture write-up has been waiting for.

Note: this **collapses** the previously-listed `start-12b-qat.sh` port-derivation tool fix —
no point fixing a foot-gun on a file we're about to delete.

---

## State at close

- **31B:** stopped at end of session (last action was the solo re-baseline run).
- **Repos:** training repo ahead by the day's commits (re-baseline JSON + 3 start-stack fixes +
  1 sweep fix); IRS untouched today.
- **New pinned anchor:** v0.23.0 digest `…6d8429e3…22ed8f`, alongside the retained v0.21.0 and
  gemma4-unified digests.

## Parked / not done today

- **nginx directory-mount** (IRS repo) — independent of the training-repo work; deferred to an
  IRS-focused moment.
- **Prefill warm-up artifact** — downgraded from "tool change" to "documented caveat": it did
  not fire today, so the harness is left unchanged and the behavior is simply noted.
- **Step 4 execution** + **full-stack native-on-v0.23.0 boot** — the clean opening for the next
  block.
