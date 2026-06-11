# Week 12, Day 1 — Does the Gemma 4 12B QAT model load on a single 24 GB card?

**Date:** 2026-06-11
**Hardware:** one RTX 3090 (24 GB), GPU 1
**Image:** `vllm/vllm-openai:gemma4-unified` (digest `sha256:e828735fba48bca2cf9701864d41693c91953394c5b1455b4668edd7563ed450`)
**Model:** `google/gemma-4-12B-it-qat-w4a16-ct`

## The question for the day

The plan for Week 12 is to build a two-tier setup: a capable model that hands off bulk work to cheaper, faster "sub-agent" models. The sub-agents have to run on the 24 GB cards we have. So Day 1 asks one thing: does the 12B QAT model actually load on a single 24 GB RTX 3090 under the new `gemma4-unified` vLLM image?

("QAT" = quantization-aware training: the model's main weights are compressed to ~4 bits, which is why we expected it to be small enough to fit. "w4a16" names that compression scheme.)

**Answer: not as configured.** It runs out of GPU memory during loading, by a small margin (~1.9 GB short). The reasons are now understood, and there are untried, cheap levers to try next session.

## What happened, in order

The load did not fail in one clean way — it failed in three different places, and each fix exposed the next problem underneath.

1. **First crash — a missing config field.** The image's multimodal setup code looked for a field (`num_soft_tokens`) that this model's config file doesn't contain, and crashed before loading any weights. We supplied the missing field as a harmless placeholder (it reserves nothing, because we'd already turned image and audio inputs off). This got us past setup.

2. **Second crash — the loader mis-handled an un-compressed layer.** The model deliberately leaves a few small input-projection layers *un*-compressed, and its config file lists them as "skip compression here." vLLM ignored that list and tried to load one of those layers (`vision_embedder.patch_dense`) as if it were compressed, then failed because the shapes didn't match. We worked around it by re-stating the skip-list using wildcard patterns, which vLLM did honor. This got us into the actual model build.

3. **Third crash — out of memory (the real finding).** With both bugs out of the way, loading reached the very last big tensor — the output projection ("lm_head") — and there was no room left. The card was already holding 23.2 GB of its 23.6 GB usable, and the lm_head needed 1.9 GB more.

The first two were software bugs in a brand-new, pre-release image, and we stepped over both. The third is a genuine memory limit, not a bug.

## Why it didn't fit (the part worth remembering)

The prediction was a comfortable fit: ~10 GB of weights on a 24 GB card. That was wrong, and the reason is specific.

The 4-bit compression applies to the model's main transformer layers, but **not** to two very large lookup tables: the input embedding and the output projection. This model has a 262,000-word vocabulary, so each of those two tables is roughly 2 GB and stays at full (16-bit) precision. That's ~4 GB the "weights are only ~10 GB" estimate didn't account for. On top of that, vLLM's default startup builds an optimized-execution working set (CUDA graphs) *while* the weights are loading, which also consumes GPU memory at exactly the wrong moment. Weights + two big uncompressed tables + startup working set pushed past 24 GB.

So the binding constraint is **not** the compressed transformer — it's the two uncompressed vocabulary tables plus the startup overhead.

## Architecture note (a correction)

Going in, the working assumption was that this multimodal model carries separate vision and audio "towers" (encoder networks) that we could switch off to save memory. That is **wrong for this model.** The 12B is "encoder-free": instead of separate encoders, it projects images and audio straight into the model with a few small linear layers (~50M parameters total, confirmed both from the weight file and the model card, which lists its vision/audio encoder size as none). There is no tower to drop. The memory-saving lever that worked on the larger 31B model does not exist here.

## Where this leaves the week

The do-or-die question has a clear, useful answer: **the model doesn't fit at the default settings, but the gap is small (~1.9 GB) and the causes are named.** That's a good place to stop — it turns "unknown" into "near-miss with specific levers left to try."

Cheapest things to try next session (single changes, one at a time):
- **Turn off CUDA-graph capture at startup** (`--enforce-eager`). This removes the startup working set entirely and could easily free more than the 1.9 GB we were short. First thing to try.
- **Lower the memory-utilization target** so vLLM reserves less and reshuffles the budget.

Out of scope and deliberately not started: hunting for a different vLLM version (that's a Week 13 task), and anything involving a second worker or the front-door router.

## State at close
- All four GPUs released and clean (1 MiB used each).
- No leftover containers.
- Two image bugs documented above for the eventual Week 13 version work.
- No results JSON written today (no successful run to record).
