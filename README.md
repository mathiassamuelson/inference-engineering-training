# RTX 3090 AI Inference Infrastructure Training

Learning **AI inference infrastructure and LLM serving** hands-on — deploying, optimizing, and
operating large-model inference on a 4× RTX 3090 workstation — and documenting the journey.

The work centers on the systems craft of inference: serving stacks (vLLM), quantization
(W4A16 QAT, FP8), tensor- and pipeline-parallelism, KV-cache and throughput/latency
characterization, multi-GPU topology (NVLink vs PCIe), and the evaluation methodology that
backs production serving decisions.

**Hardware:** 4× NVIDIA RTX 3090 (96 GB total VRAM) · Ubuntu 24.04 · CUDA 12.x · two GPUs
NVLink-paired, the other two on PCIe.

## Two repositories

This is the **data** repository — results, journals, and captures. The **toolchain** lives in a
separate repo:

- **[`rtx3090-ai-training`](https://github.com/mathiassamuelson/rtx3090-ai-training)** (this repo) —
  results, per-week journals, captured measurements, and the training plan.
- **[`rtx3090-ai-training-tools`](https://github.com/mathiassamuelson/rtx3090-ai-training-tools)** —
  benchmarking and evaluation tools, plus the bundled eval inputs (prompts, probes, rubrics).

The split keeps outputs and code separate: tools are versioned in the tools repo, run from here,
and write their results here.

## Run convention

Check out both repos side by side, and run tools **from this repo** so results land here:

```bash
cd ~/work/rtx3090-ai-training                     # CWD = this (data) repo
T=~/work/rtx3090-ai-training-tools                # the tools repo checkout

python3 "$T/tools/throughput_sweep.py" \
    --backend vllm-openai --endpoint http://localhost:8000 \
    --results-dir phase-3-optimization-and-quantization/week-14/results
```

Result files are self-describing (model name and run config folded into the filename), and each
records the **tools repo's** git SHA for provenance — so a result committed here is always
traceable to the exact tool revision that produced it, even though it was run from this repo. See
the tools repo's README for setup, the provenance model, and the full tool list.

## Repository structure

```
phase-N-name/        organized by training phase
  week-NN-*/          per-week journals, results, and captures
docs/
  training-plan.md    the full curriculum
  linkedin/           published write-ups
  compendiums/        reference deep-dives
```

The curriculum, current focus, and findings live in `docs/training-plan.md` and the per-week
journals — not here — so this README stays a stable description of the repository rather than a
status page.

## Quick start

```bash
# clone both repos side by side
git clone https://github.com/mathiassamuelson/rtx3090-ai-training.git
git clone https://github.com/mathiassamuelson/rtx3090-ai-training-tools.git

# set up this repo's environment (Phase-1 provisioning stack)
cd rtx3090-ai-training
./setup.sh
source ~/ai-inference/bin/activate
```

The tools repo has its own slim environment recipe; both target the shared `~/ai-inference` venv.

## Connect

- LinkedIn: https://www.linkedin.com/in/mathiassamuelson/

## License

MIT License — feel free to use any code or approaches documented here.
