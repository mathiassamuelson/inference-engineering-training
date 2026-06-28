# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is the **data** repository for an AI inference infrastructure training program — results,
per-week journals, captured measurements, and the curriculum. It is one of two side-by-side repos:

- **`rtx3090-ai-training`** (this repo, "R") — results, journals, captures, `docs/training-plan.md`.
- **`rtx3090-ai-training-tools`** ("T") — the benchmarking/eval toolchain plus bundled eval inputs
  (prompts, probes, rubrics).

Tools are versioned in T, **run from here (R)**, and write their results here. Do not add tools to
R, and never let a tool write its output into T.

## Run convention

Check out both repos side by side. Run a tool from R so results land in R:

```bash
cd ~/work/rtx3090-ai-training            # CWD = R (this repo)
T=~/work/rtx3090-ai-training-tools       # T checkout

python3 "$T/tools/throughput_sweep.py" --backend vllm-openai \
    --endpoint http://localhost:8000 \
    --results-dir phase-3-optimization-and-quantization/week-14/results
```

- Result paths resolve against the **CWD** (R). A relative `--results-dir` writes into R.
- Provenance recorded in each result is **T's git SHA** (anchored to the tool file via
  `tools/provenance.py`, not the CWD). A result committed in R is therefore traceable to the exact
  tool revision that produced it.
- Result filenames are self-describing (model name + run config folded in) so runs against
  different models/configs never silently overwrite.

See T's README for the full tool list, setup, and the provenance model.

## Session discipline

- **Commit before running** any results-writing experiment. Verify a clean tree first
  (`git status --porcelain` empty) — a dirty tree contaminates the recorded git SHA (the
  "dirty-tree trap").
- **`git pull` before pushing.** R has two writers (the inference box and the Mac); use
  `git pull --rebase` on R before pushing inference-side commits. `git pull --ff-only` is the
  pre-session gate on both repos.
- **One experiment per session/boot; predict before measuring.** Change one variable at a time;
  record the prediction and the deviation.
- **Historical journals are not rewritten.** They were accurate when written; only live reference
  docs (this file, READMEs) are updated to current truth.

## Serving and endpoint footguns

- **Pinned image:** `vllm/vllm-openai:v0.23.0`
  (`sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f`). It loads all three
  production models (31B-QAT orchestrator, both 12B-QAT workers, BF16 parents) with **zero**
  per-model workarounds. Pre-v0.23 scaffolding (source patches, `--hf-overrides` quantization
  blobs, the temporary `start-12b-qat.sh`) is retired — do not reintroduce it.
- **Chat endpoint only.** Use `/v1/chat/completions`. Raw `/v1/completions` produces degenerate
  (token-repetition) output on the Gemma 4 instruction-tuned models.
- **The `--endpoint`/`--base-url` `/v1` suffix differs per tool** — passing the wrong form 404s
  silently:
  - `throughput_sweep.py --endpoint http://host:port` — **no** `/v1`; the script appends
    `/v1/models` and `/v1/completions` itself.
  - `rca_quality_probe.py --base-url http://host:port/v1` — **include** `/v1`; the script appends
    only `/chat/completions`.
  - `vllm-bringup-checks.sh --host H --port P` — takes host/port separately and builds the base.
- **Concurrency is one integer per invocation** for `throughput_sweep.py` (`--concurrency N`); run
  separate invocations to sweep concurrency levels.
- **Verify GPU placement empirically**, never trust launcher intent: `vllm-bringup-checks.sh`
  performs the UUID → PID → physical-GPU join. The `start-vllm.sh` intent echo states *intended*
  placement only.

## Launching models

`start-vllm.sh` (in T) has role presets; explicit flags override, and there is no silent zero-arg
launch (no role and no `--model` prints usage and exits):

```
orchestrator   31B-QAT  TP=2  GPUs 0,2 (NVLink pair)   port 8000  MML 131072  util 0.95
worker1        12B-QAT  TP=1  GPU 1                    port 8001  MML 131072  util 0.90
worker2        12B-QAT  TP=1  GPU 3                    port 8002  MML 131072  util 0.90
```

`start-stack.sh` (in T) boots the whole multi-tier stack (both workers + orchestrator, optionally
the nginx front door) with time-to-healthy probing and a `teardown` verb.

## Tooling conventions (when editing tools in T)

- **Multi-model by default.** Accept the model name as a CLI argument with a sensible default;
  never hardcode it. Propagate model identity into request payloads, console headers, summary
  tables, JSON metadata, and the default output filename.
- **Self-describing outputs.** Fold model name and run config (e.g. concurrency, parallelism tag)
  into the default result filename so runs do not overwrite each other.
- Do not put model-specific constants (context limits, tensor dims) in output metadata unless they
  are computed from the actual model under test.

## Repository structure

```
phase-1-foundation/                    Weeks 1-4
phase-2-production/                     Weeks 5-8 (... )
phase-3-optimization-and-quantization/ Weeks 11-16 (current)
  week-NN-*/                           per-week journals, results, captures
docs/
  training-plan.md                     the full 28-week curriculum (authoritative for phase/week scope)
  linkedin/                            published write-ups
  compendiums/                         reference deep-dives
setup.sh, requirements.txt            this repo's environment recipe (Phase-1 provisioning stack)
```

The curriculum, current focus, and findings live in `docs/training-plan.md` and the per-week
journals — consult those for status and scope rather than assuming from this file.

## Hardware context

4× NVIDIA RTX 3090 (96 GB total VRAM), Ubuntu 24.04, CUDA 12.x. GPUs 0+2 are NVLink-paired
(~100 GB/s); GPUs 1+3 are on PCIe 3.0 x1 — so tensor-parallel pairs belong on 0+2, and the x1 cards
host single-GPU workers (no inter-GPU traffic, so the slow link never enters the critical path).
