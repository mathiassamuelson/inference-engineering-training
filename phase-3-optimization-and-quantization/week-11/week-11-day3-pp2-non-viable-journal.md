# Week 11 — Day 3: PP=2 is non-viable for FP8 31B Dense on two GPUs

**Date:** 2026-06-01
**Config family:** FP8 31B Dense (`RedHatAI/gemma-4-31B-it-FP8-block`), vLLM 0.21.0, text-only, GPUs 0 & 2 (NVLink pair), Ubuntu 24.04 / CUDA 12.6.
**Goal as planned:** characterize PP=2's KV pool empirically, then run the c=1 sweep to complete the TP=2 vs PP=2 single-request comparison.
**Outcome:** PP=2 cannot hold a usable context window at the held-constant config. The single-request sweep was abandoned — there is nothing meaningful to overlay against TP=2. The non-viability *is* the finding. Pivot to PP=4 deferred to Day 4.

---

## Held-constant config (carried from Day 2)

- vLLM **0.21.0**, pinned digest `sha256:a230095847e93bd4df9888b33dab956fa9504537b828a23657d2b26fed57b5c9` (verified via `RepoDigests`).
- **text-only** (`--limit-mm-per-prompt '{"image":0,"audio":0,"video":0}'`, script default; vision tower skipped — confirmed "running in text-only mode" in the boot log).
- **graphs ON**, `--gpu-memory-utilization 0.95`, `--kv-cache-dtype auto` (→ BF16 on SM 8.6), `--max-num-batched-tokens 4096`.

---

## What happened

### Boot at matched MML 33024 — failed at KV init

The PP=2 launch at the Day-2-matched MML 33024 did not start serving. The engine died in `_check_enough_kv_cache_memory` with:

> To serve at least one request with the model's max seq len (33024), 3.22 GiB KV cache is needed, which is larger than the available KV cache memory (1.95 GiB). Based on the available memory, the estimated maximum model length is 4624.

So at util 0.95, the binding pipeline stage exposes only **1.95 GiB** of KV, while a single full-window (33024-token) sequence needs **3.22 GiB**. vLLM's own estimate of the largest serviceable context: **~4624 tokens**.

### Re-probe at MML 4096 — clean readout

Re-launching at a fit-able MML produced a successful boot and the real per-stage pool:

```
Available KV cache memory (PP0 / GPU 0):  1.95 GiB     <- binding stage
GPU KV cache size:                        4,626 tokens
Maximum concurrency for 4,096 tokens:     1.13x
```

The **4,626-token** pool agrees with the crash's **4624** prediction to within rounding — strong internal consistency, two independent code paths arriving at the same ceiling.

### Placement confirmed (physical GPUs, not inferred)

`nvidia-smi` compute-apps joined to the GPU table on `gpu_uuid`:

| PID    | gpu_uuid (short) | physical index | mem used |
|--------|------------------|----------------|----------|
| 341054 | `…b56f`          | **GPU 0**      | 20636 MiB |
| 341055 | `…d7ba`          | **GPU 2**      | 21886 MiB |
| —      | `…6398`          | GPU 1          | 1 MiB (idle) |
| —      | `…3d2c`          | GPU 3          | 1 MiB (idle) |

Both stages landed on the NVLink pair (0 & 2); the PCIe-x1 cards (1 & 3) stayed out. No accidental x1 involvement. The earlier `local_rank 0/1 → physical 0/2` inference held.

---

## Why PP=2 craters: the embedding doesn't shard

**What the embedding and LM head are.** A token enters the model, the **embedding** table turns it into a vector, that vector flows through all the transformer layers, and comes out as a single hidden vector — the model's internal representation of "what comes next," but in its own coordinate space, not in words. The **LM head** (language-model head) is the final projection that maps that hidden vector to one score (a *logit*) per token in the vocabulary; softmax over those logits gives the distribution the next token is sampled from. Both are `hidden_dim × vocab_size` matrices, so with Gemma 4's 256K vocabulary they are the two largest non-layer tensors in the model — the embedding maps vocab → hidden on the way in, the head maps hidden → vocab on the way out. (In tied-embedding models these are literally the same weights reused; vLLM's handling of that tie under PP is a detail that doesn't change the conclusion below — a 256K-wide tensor sits un-sharded on a stage either way.)

Under **TP**, those two vocab-sized tensors shard across both GPUs, so each card carries half. Under **PP**, they do not — the full embedding lands whole on the first stage and the LM head whole on the last. That unsharded vocab tensor inflates a single stage's weight footprint and crushes its KV floor. Because a pipeline's usable pool is bounded by the **minimum across stages (PP ranks)**, that one starved stage caps the whole pipeline.

The numbers, TP=2 (Day 2) vs PP=2 (Day 3), identical util 0.95:

| metric                  | TP=2 (Day 2) | PP=2 (Day 3) |
|-------------------------|--------------|--------------|
| weights / binding GPU   | 15.85 GiB    | 17.06 GiB (stage 0) |
| available KV / binding GPU | 4.04 GiB  | 1.95 GiB     |
| KV pool                 | ~41,300 tok  | 4,626 tok    |
| max single-seq context  | ~55,400 tok  | ~4,624 tok   |
| max-conc at ladder      | 1.25x @ 33024 | 1.13x @ 4096 |

PP=2's serviceable context is **~12× smaller** than TP=2's on the same hardware at the same util. There is no MML at which PP=2 supports the matched 512–32768 ladder — an 8192-token prompt alone exceeds its entire pool.

---

## Where the 24 GiB goes (and why KV is the part that moves)

Weights + KV only account for ~19.9 GiB (TP) and ~19.0 GiB (PP) of each card's 24 GiB. The remaining ~4–5 GiB is not unaccounted — it's spoken for, just not in the two headline numbers. Reconstructing the full per-GPU budget:

| component | TP=2 / GPU | PP=2 stage 0 / GPU | notes |
|-----------|-----------:|-------------------:|-------|
| total (RTX 3090)              | 24.00 | 24.00 | physical VRAM |
| 5% safety margin (util 0.95)  |  1.20 |  1.20 | vLLM never allocates above 0.95 → 24 × 0.05 |
| model weights                 | 15.85 | 17.06 | measured at load |
| CUDA-graph capture            | ~0.65 |  0.65 | logged for PP; for TP estimated at the same magnitude |
| non-torch + activation        | ~2.26 | ~3.14 | CUDA context, NCCL/comm buffers, peak activation — **by difference** |
| **KV cache (residual)**       | **4.04** | **1.95** | whatever's left inside the 0.95 budget |

(Each column sums to 24.00.) Two figures are derived, not directly logged: the **CUDA-graph** number for TP (we only captured PP's, but the PP value is corroborated two ways — the explicit "Estimated CUDA graph memory: 0.65 GiB" line *and* the util-equivalence note, where 0.95 → 0.9226 implies (0.95 − 0.9226) × 24 = 0.66 GiB), and the **non-torch + activation** row, which is computed as the residual-of-residuals (everything in the 0.95 budget not otherwise named). So treat that last row as a bucket, not a precise measurement.

**The conceptual point for future-me: KV is the residual, not a budgeted line item.** vLLM measures everything else *first* — weights, CUDA-graph capture, CUDA context, NCCL buffers, peak activation during a profiling forward pass — and then hands KV whatever is left under the 0.95 ceiling. That inverts the intuition: KV doesn't shrink because "KV got more expensive," it shrinks because *something else grew*. So the right question is never "why is the KV pool small," it's "what ate the budget before KV got its turn." Under PP=2, two things did: the un-sharded embedding/LM-head made weights +1.21 GiB heavier, **and** the non-torch/activation bucket grew ~0.9 GiB (pipeline staging buffers being the prime suspect). Both came straight out of the residual, which is why KV fell −2.09 GiB — more than the weight increase alone would predict. That ~0.9 GiB excess is the same number flagged in caveat 1 below.

---

## Open caveats (flagged, not smoothed)

1. **The weight delta doesn't fully reconcile the KV deficit.** Stage 0 weights are +1.21 GiB vs TP (17.06 vs 15.85), but available KV is −2.09 GiB (1.95 vs 4.04). The ~0.9 GiB difference is the extra non-torch/activation overhead PP carries beyond its heavier weights — see "Where the 24 GiB goes" above; pipeline staging buffers are the likely source, but it's a residual-of-residuals figure, not a directly measured one. The **direction and primary cause (un-sharded embedding/LM-head) are unambiguous**; the exact attribution of that last ~0.9 GiB is open and not worth chasing given PP=2 is being abandoned.

2. **Only PP0's available-KV line was captured.** The "minimum rank = PP0" conclusion rests on PP0's 1.95 GiB being the figure that drove the 4,626-token pool (which it matches via the crash prediction). PP1's `Available KV cache memory` line was not in the captured output, so the stage-to-stage asymmetry is inferred from the binding figure, not directly from both stages' lines side by side. Confident enough for the conclusion; noted for completeness.

3. **A total-memory misread, corrected.** `nvidia-smi` showed GPU 2 carrying ~1.25 GiB more *total* memory than GPU 0 (21.9 vs 20.6 GiB), which initially looked like GPU 2 being the heavier/binding stage. The boot log is authoritative and says otherwise: the binding **available-KV** floor is PP0 / GPU 0 at 1.95 GiB. Total resident memory ≠ available KV; they were conflated momentarily and the log corrected it.

---

## Conclusion

On two GPUs, for 31B Dense at FP8, **tensor parallelism is the only practical option.** PP=2 is structurally non-viable here because the un-sharded embedding starves one stage's KV pool, capping serviceable context at ~4,624 tokens — far below any realistic workload and ~12× below TP=2. This is a clean deployer-relevant result and belongs in the record, but it does not carry a standalone publication on its own.

The interesting remaining question is whether **PP=4** — the only config that actually halves per-GPU weights again (~7.9 GiB/GPU) and roughly halves per-stage layer count — recovers a usable KV pool, making TP=2 vs PP=4 the real "two practical configs for this model on this box" comparison. That requires putting the PCIe-x1 cards (GPUs 1 & 3) into the pipeline, which PP tolerates far better than TP would, but which makes decode latency the variable to watch. Deferred to Day 4.

---

## Deliverables this session

- This journal: `week-11-day3-pp2-non-viable-journal.md`.
- No sweep JSON committed — PP=2 produced no comparable single-request curve; the failure-mode and pool characterization above are the artifact.
- TP=2 Day 2 anchors remain the comparator for whatever Day 4's PP=4 produces.
