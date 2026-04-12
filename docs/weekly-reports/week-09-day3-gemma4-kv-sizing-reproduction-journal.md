# Week 9 Day 3 — Gemma 4 KV Sizing Reproduction

## Overview

Day 3 was planned as a vLLM KV allocation investigation: read the source, run an empirical `--max-model-len` sweep, and cross-reference against other hybrid-attention implementations to determine whether vLLM is exploiting `sliding_window` / `layer_types` when sizing the KV cache for Gemma 4 26B MoE. Motivating observation from Day 2: vLLM appeared to need ~5.5× more KV memory per token of capacity than llama.cpp (120 KB/token vs 22 KB/token).

The plan changed on first contact. Within the first tool call, search surfaced [vllm-project/vllm#39133](https://github.com/vllm-project/vllm/issues/39133), filed by @ormandj on 2026-04-07 against Gemma 4 31B Dense on identical-class hardware (2× RTX 3090, TP=2, same `cyankiwi` AWQ-4bit checkpoint family). The issue asks the exact question Day 3 was set up to answer. Status: no assignee, no labels, no linked PRs, no comments as of 2026-04-12.

Rather than duplicate ormandj's work, Day 3 pivoted to a focused reproduction-and-contribution: confirm the same symptom on the 26B MoE variant, collect empirical data #39133 doesn't yet have, post observations as a comment on the existing issue. Source-code investigation was dropped as unnecessary duplication.

Posted comment: https://github.com/vllm-project/vllm/issues/39133#issuecomment-4232552320

## Sequence of events

**Version pinning.** Before any source-code or empirical work, pinned the exact vLLM commit in the running container so any claim in the comment would be anchored to the code that produced our benchmark numbers. `pip show vllm` inside the `vllm/vllm-openai:gemma4` container reports `0.18.2rc1.dev73+gdb7a17ecc`; `_version.py` exposes `__commit_id__ = 'gdb7a17ecc'`. Image digest `sha256:0cb12dc964e1...` confirmed identical to the image used for Day 1 and Day 2 benchmarks. The image was built 2026-04-02 (Gemma 4 release day), which aligns with vLLM's official [Gemma 4 blog post](https://vllm-project.github.io/2026/04/02/gemma4.html) recommending this exact image.

**Reading #39133.** Ormandj's report is careful and technically solid. Captures the backend-forcing log line (`Gemma4 model has heterogeneous head dimensions... Forcing TRITON_ATTN backend`) that our runs also produce. Reports 25,200 tokens of KV capacity at `max_model_len=131072` with `gpu_memory_utilization=0.96` on 31B dense. Frames the question carefully: "I am not asserting that vLLM is incorrect — I am reporting the exact observations and asking whether they match the current design." The secondary `fp8_e5m2` gate observation was orthogonal to our question and not reproduced.

**26B MoE architectural pull.** Pulled `text_config` from the cached model on disk. Key fields:
- 30 layers, 25 SWA + 5 full, interleaved 5:1 with final layer full
- `num_key_value_heads: 8` (SWA), `num_global_key_value_heads: 2` (global)
- `head_dim: 256`, `global_head_dim: 512`
- `sliding_window: 1024`
- `attention_k_eq_v: true` (per model card: "global layers feature unified Keys and Values")

The additional global-layer KV compressions (`num_global_key_value_heads: 2` and `attention_k_eq_v: true`) don't appear in ormandj's 31B dense parameters. Whether they're absent from the 31B config or simply not reported is unknown — this is a genuinely new data point from the 26B MoE variant.

**Sweep.** Four runs at `--max-model-len` ∈ {8192, 32768, 131072, 262144}, fixed `--gpu-memory-utilization 0.90`. Captured the startup log lines for KV pool size and KV cache size in tokens.

| `--max-model-len` | Available KV (per GPU) | GPU KV cache size |
|---:|---:|---:|
| 8,192 | 10.93 GiB | 95,472 tokens |
| 32,768 | 10.92 GiB | 95,440 tokens |
| 131,072 | 10.92 GiB | 95,456 tokens |
| 262,144 | 10.93 GiB | 95,472 tokens |

Across a 32× range in `max_model_len`, KV token capacity varies by 32 tokens — exactly two paged-attention blocks of `block_size=16`. The pool size is constant, as expected; per-token cost is constant; reducing `max_model_len` buys essentially no KV headroom.

## Per-token math and interpretation

Observed per-token cost: 10.93 × 2³⁰ / 95,472 ≈ 122,925 bytes/token per GPU.

For one attention layer under TP=N, per-GPU per-token KV footprint is `(num_kv_heads / TP) × head_dim × K_V_factor × sizeof(dtype)`, where `K_V_factor = 1` for layers with `attention_k_eq_v` and `2` otherwise. For 26B MoE at TP=2, BF16:

- SWA layer: `(8/2) × 256 × 2 × 2 = 4,096 bytes/token`
- Global layer: `(2/2) × 512 × 1 × 2 = 1,024 bytes/token`

Three candidate interpretations:

1. **Full hybrid-attention exploitation** (SWA capped at 1024, globals at max_model_len, `attention_k_eq_v` honored): predicts ~2.1M tokens at M=262144. Observed is 25× lower. Rejected.
2. **Heterogeneous layers honored, no SWA cap** (25 × 4,096 + 5 × 1,024 = 107,520 bytes/token): predicts ~109K tokens. Observed 14% lower. Requires a block-overhead factor not otherwise evident.
3. **All 30 layers treated as SWA-shaped, no SWA cap** (30 × 4,096 = 122,880 bytes/token): predicts ~95,500 tokens. Observed matches within 0.04% — within precision of the 2-decimal-place pool-size report.

Interpretation 3 is tight enough to be the primary hypothesis. It's consistent with vLLM's KV manager reading only top-level `num_key_value_heads` and `head_dim` from `text_config` and applying them uniformly to all layers, sized at `max_model_len`. Under this interpretation, four per-layer fields in the Gemma 4 config are not consulted during KV pool sizing: `sliding_window` × `layer_types`, `num_global_key_value_heads`, `global_head_dim`, and `attention_k_eq_v`.

This is a broader finding than #39133 originally framed. The SWA windowing question is one of four; the full fix may require reading layer-type-specific shape information rather than only uniform top-level fields.

## Contribution to #39133

Posted a comment covering: environment pinning, 26B MoE architectural params (highlighting fields absent from ormandj's 31B dense report), the four-point sweep table, the three-interpretation per-GPU per-token analysis, and the "four fields not yet consulted" summary. Framed as confirmatory reproduction rather than a new report. Explicitly matched ormandj's non-accusatory tone in the closing paragraph.

Comment archived at `results/issue-39133-comment-26b-moe-reproduction.md` for reference; live version at https://github.com/vllm-project/vllm/issues/39133#issuecomment-4232552320.

## Week 9 close-out

Week 9 reaches a natural stopping point at Day 3 rather than continuing through the originally planned Day 4 concurrent benchmarking and Day 5 full report.

The Day 2 single-request findings — layer splitting pulling ahead of TP=2 at long context for both prefill and decode — are suspect in light of #39133. vLLM's oversized per-layer KV allocation plausibly penalizes TP configurations more than layer-splitting configurations: TP pays the cost in per-step bandwidth through the attention kernel and all-reduce, while layer splitting pays it in VRAM only. The crossovers observed at ~8K (decode) and ~32K (prefill) may be artifacts of the bug rather than true architectural properties of this hardware. Publishing that conclusion now would either age badly or require caveats heavy enough to undermine the writeup. Running Day 4 concurrent benchmarking against the same vLLM build has the same problem at larger scale — the numbers would be real, but the interpretation would be provisional.

Week 9 therefore closes with the Day 2 single-request data and the #39133 contribution as its concrete outputs. The TP-vs-layer-splitting investigation is **paused, not abandoned**. When #39133 resolves, the existing benchmark scripts (`tools/throughput_sweep.py` at schema v2) and methodology are ready to re-run against the fixed vLLM; the before-and-after comparison becomes a finding in its own right, arguably more interesting than the original plan because it concretely demonstrates how a single framework bug can invert architectural conclusions. That re-test is a natural LinkedIn Pulse candidate if the result is publishable.

Staying on Gemma 4 for Week 10 preserves the context needed for that eventual re-test — model weights, benchmark scripts, methodology, and mental familiarity all carry forward without a context-switch tax.

## Week 10 pivot

Week 10 shifts focus from runtime-layer optimization to production-deployment concerns: Triton Inference Server with the vLLM backend, fronted by nginx, with statmon-ai as the driving client application.

Rationale:

- **vLLM backend is chosen for known-working Gemma 4 compatibility.** #39133 affects capacity, not correctness; the bug is transparent to the production-deployment work Week 10 tackles.
- **Production concerns are a legitimate and under-covered side of AI infrastructure.** The original curriculum leaned heavily toward runtime optimization; this broadens scope into TLS termination, reverse-proxy config, observability, API keys, and per-token metering. These are foundational for any real deployment.
- **statmon-ai as a real client application forces questions synthetic benchmarks can't answer.** The interface contract between statmon-ai and the inference service will be OpenAI-compatible, allowing both sides (the Triton stack and the statmon-ai refactor) to develop independently.
- **Sticking with Gemma 4 avoids the mental tax** of introducing a new model family mid-program, and keeps the Week 9 re-test accessible when #39133 resolves.

Week 10 scope: **observability first** (Prometheus + Grafana against Triton's built-in metrics and nginx access logs), then **HTTPS termination and reverse-proxy config via nginx**. API keys and per-token billing/metering move to Week 11.

The original Week 10 plan (Triton + TRT-LLM framework comparison) is preserved and pushed forward — same treatment as the Week 8 Gemma 4 insertion gave to the original Triton content.

Operational note for Week 10: default nginx `proxy_buffering on` is incompatible with streaming responses from OpenAI-compatible chat completions. Observability should make buffering misconfigurations visible early — first-token latency approaching full-response latency is the signature.

## #39133 status as Week 9 closes

- Reported upstream with two-architecture reproduction (31B dense via ormandj, 26B MoE via our contribution)
- No assignee or linked PR as of 2026-04-12; watch item rather than blocker
- No vLLM config adjustment discovered that mitigates the issue. The `--kv-cache-dtype fp8_e5m2` workaround mentioned in #39133 hits a compressed-tensors gate and is unavailable for INT4 checkpoints; `fp8_e4m3` hits a Triton SM 8.6 limitation on Ampere. BF16 KV is the only viable option on this hardware until upstream fix lands.

## Methodology notes

**Check for existing issues before starting novel investigation.** A 2-minute web search saved what would have been hours of duplicative source-code reading. The habit of searching the upstream issue tracker as step zero of any investigation is worth keeping.

**Tight per-token arithmetic on a single measurement is highly discriminating** when the candidate interpretations are architecturally grounded. The 0.04% match between observed 122,925 and predicted 122,880 bytes/token is more informative than a noisy multi-point fit would have been, because each predicted value is an exact architectural calculation with no free parameters.

**"Per GPU" instead of "per rank" for external communication.** "Rank" is standard vocabulary in distributed-ML internals but reads as jargon to a broader audience. Worth maintaining this distinction: internal notes can use whichever is most precise; public posts should prefer plainer language.

**Don't cite metrics in public posts without understanding their computation.** The initial sweep table included vLLM's reported "Maximum concurrency for X tokens per request" figure, which turned out to be a ratio vLLM computes internally via logic we haven't traced. Removing it was correct — even if it's fine, citing a number we can't explain would be sloppy. If a metric matters to an argument, trace it; if it doesn't, omit it.
