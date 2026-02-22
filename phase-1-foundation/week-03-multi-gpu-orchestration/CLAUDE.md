# CLAUDE.md — Week 3: Multi-GPU Orchestration

## Overview

This week measures how 4x RTX 3090s communicate over PCIe and quantifies the
cost/benefit of each multi-GPU parallelism strategy for LLM inference.

## Experiments

| Script | Experiment | Purpose |
|--------|------------|---------|
| `gpu_topology_benchmark.py` | 1 — GPU Topology & Communication Baseline | NVLink/PCIe bandwidth, peer access, latency, ring all-reduce |
| `data_parallel_scaling.py` | 2 — Data Parallelism Throughput Scaling | Independent replicas across GPUs; PCIe x1 vs x16 slot penalty |
| `pipeline_parallel_benchmark.py` | 3 — Pipeline Parallelism Performance | Layer-split overhead across 1/2/4 GPUs over PCIe |
| `cuda_streams_benchmark.py` | 4 — CUDA Streams & Async Execution | Stream serialization, compute/transfer overlap, sync cost |

## Running Experiments

```bash
source ~/ai-inference/bin/activate

python3 gpu_topology_benchmark.py
python3 data_parallel_scaling.py
python3 pipeline_parallel_benchmark.py
python3 cuda_streams_benchmark.py
```

All scripts print results to stdout. No CSV output for this week.

## Hardware Context

- GPU 0: PCIe x16 slot
- GPUs 1–3: PCIe x1 slots (bandwidth-limited for H2D/D2H transfers)
- No NVLink between RTX 3090s — all inter-GPU traffic is over PCIe

## Key Findings (from this week)

- **Data parallelism** is the best strategy for models that fit on one GPU.
  No inter-GPU communication at inference time; scales linearly.
- **Pipeline parallelism** adds per-stage latency due to activation transfers
  over PCIe. Only justified when the model cannot fit on a single GPU.
- **PCIe x1 slots** do not meaningfully hurt on-chip compute throughput —
  the penalty only appears when transferring large tensors between host and device.
- **Compute/transfer overlap** (Experiment 4C) requires pinned (page-locked)
  memory. Regular pageable memory cannot be transferred asynchronously.
- **Over-synchronization** (`torch.cuda.synchronize()` inside tight loops) is
  a common performance bug. Synchronize at batch boundaries, not per-operation.

## Known API Gotchas

- `torch.cuda.get_device_properties(i).total_memory` — correct attribute name.
  (`total_mem` does not exist and raises `AttributeError`.)

## Git Workflow

- Branch per week: `week-01`, `week-02`, `week-03`, etc.
- Commit messages: imperative subject line (≤72 chars), blank line, descriptive body.
- Merge week branch → `main` when the week is complete.

```bash
git checkout -b week-03
git add <file>
git commit   # write professional message
git push origin week-03
# when done:
git checkout main && git merge week-03 && git push origin main
```
