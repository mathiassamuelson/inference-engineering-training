# Week 8 Summary — Gemma 4 day-1 deployment: 31B Dense, 26B MoE, two frameworks

**Dates:** 2026-04-03 → 2026-04-07 (Days 1–4)
**Models:** `ggml-org/gemma-4-31B-it-GGUF` Q8_0 (dense), `ggml-org/gemma-4-26B-A4B-it-GGUF` Q8_0 (MoE, 25.2B total / 4B active), `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`
**Hardware:** 4× RTX 3090, NVLink pair GPUs 0+2, PCIe-only pair GPUs 0+1 used as the topology contrast
**Frameworks:** llama.cpp (b8660 → post-fix builds), vLLM (0.19.0 attempt Day 1; `vllm/vllm-openai:gemma4` image, 0.18.2rc1 dev build, Day 4)

## TL;DR

Week 8 abandoned the planned Triton curriculum to deploy Google's Gemma 4 family within 24 hours of release — a deliberate pivot into day-1 deployment reality (Triton was pushed to Week 9). Four sessions produced: a reported-and-fixed llama.cpp segfault; a corrected 31B Dense throughput baseline after a benchmark artifact was root-caused to server-side prefix caching; a motherboard-topology finding (no P2P DMA off the NVLink bridge — CUDA silently stages through host memory); a dense-vs-MoE head-to-head where the 26B-A4B MoE won 4.6–4.7× on decode and fit the full 262K context the dense model couldn't; and a six-failure vLLM bring-up that ended the FP8-on-Ampere road and landed on AWQ-INT4. Two predictions missed in instructive ways, and both misses are preserved below. Concurrency was **not** measured this week — every multi-user number here is engine-reported capacity math.

## The 31B Dense arc (Days 1–2)

Deployed the 31B Dense Q8_0 across the NVLink pair via llama.cpp layer splitting on release day +1. Startup exposed the architecture: hybrid attention with 50 sliding-window layers (1,024-token window) and 10 global layers, and fewer KV heads (4 vs 16) on exactly the global layers that store the most tokens. Context auto-shrunk from the 262K architectural window to 104,704 tokens to fit 2× 24 GB.

**The segfault.** The model crashed on prompts above ~5,400 tokens — binary-searched to 5,482-works / ~5,600-crashes, reproduced on single- and multi-GPU (ruling out pipeline parallelism as the cause). Reported upstream as a day-1 Gemma 4 support bug; the maintainers fixed it overnight, and Day 2 verified the fix end-to-end with no workarounds.

**Thinking mode off.** Gemma 4 ships with thinking on by default; a modified Jinja template (`gemma4-no-think.jinja`) that strips it cut response time 3.6× (8.3 s → 2.3 s) at identical decode speed — the savings are entirely from not generating ~200 hidden reasoning tokens.

**vLLM FP8 on Day 1: did not start.** Dynamic FP8 weight-only quantization loaded 16.47 GiB of weights per GPU and OOM'd during warmup on every attempted configuration. On 24 GB cards, vLLM's extra ~1 GiB of fixed overhead versus llama.cpp (no CPU-mapped embedding table, NCCL buffers, sampler state) was the difference between barely-fits and doesn't-fit.

## The benchmark-artifact correction (Day 2) — methodology headline

The first clean-looking sweep contained a reproducible anomaly: the 4K-prompt row reported prefill ~40% below its neighbors, exactly reproducible on rerun. Root cause: **llama-server's slot cache was reusing the common prefix across the benchmark's structurally similar prompts**, so `timings.prompt_n` (tokens actually prefilled — a work counter) diverged 2:1 from `usage.prompt_tokens` (true prompt size), and fixed per-request overhead was being amortized over half the assumed work.

Two traps inside the fix are worth keeping:

- `"cache_prompt": false` is **silently dropped** by llama-server's OpenAI-compatible endpoint — it only works on the native `/completion` endpoint. No error, no warning.
- The working defeat is a UUID nonce prepended to every prompt, which zeroes the longest-common-prefix match and forces cold prefill.

**Corrected 31B baseline (NVLink pair, single stream):** prefill rises from ~862 tok/s at 519 tokens to a **~1,170 tok/s plateau from 14K onward** — ~18% higher than the artifact-laden measurement suggested. Decode 23.9 → 20.4 tok/s across 519 → 28,085 prompt tokens, a 14.6% drop over a 54× context span: weight-bandwidth-bound everywhere on the curve.

## NVLink vs PCIe, and the topology finding (Day 2)

Swapping the pair from 0+2 (NVLink) to 0+1 (PCIe) under layer-split pipeline parallelism:

- Dense **decode: unaffected** (as predicted — one small activation handoff per forward pass).
- Dense **prefill: −21% at 28K, peaking −30% mid-range** — an order of magnitude worse than the ~1.5% predicted from bandwidth math (prediction miss #1, below).

`nvidia-smi topo -p2p r` explained it: every non-NVLink GPU pair on this motherboard reports `CNS` — peer-to-peer unsupported — so CUDA silently staged transfers through pinned host memory (GPU → RAM → GPU), paying two hops plus synchronization per transfer, ~14 chunked transfers for a 28K prefill. The interpretation that stuck: **Week 7's "NVLink is a precondition for TP on this hardware" was really about peer access existing at all, not about bandwidth.** `nvidia-smi topo -p2p r` is the pre-purchase diagnostic.

## Dense vs MoE head-to-head (Day 3)

Same sweep, same pair, same Q8_0 quantization — only the model changed. The 26B-A4B MoE is structurally a half-depth, half-width sibling (30 vs 60 layers, same 5:1 SWA/global ratio) with 128-expert MoE FFN blocks, 8 routed per token.

| Axis | 31B Dense | 26B-A4B MoE |
|---|---|---|
| Decode (NVLink) | 23.9 → 20.4 tok/s | 112 → 94 tok/s — **flat 4.6–4.7× at every context length** |
| Prefill (NVLink) | ~1,170 tok/s plateau | ~4,200–4,276 tok/s — 2.6× at short context growing to 3.6× at 28K |
| Max context that fit | 104,704 (auto-shrunk), 45.7 GB | **full 262,144**, 36.3 GB with 11.4 GB free |
| PCIe prefill penalty | −21% at 28K | −24% at 28K (−14–30% across the curve) |
| PCIe decode penalty | noise-level | **consistent ~−6% at every context length** (new finding) |

Interpretations, marked as such:

- **The flat decode multiplier** is the signature of a pure weight-bandwidth mechanism; 4.7× measured against a 7.7× theoretical ceiling (active/total parameter ratio) ≈ 61% realized, diluted by costs MoE doesn't reduce (KV traffic, global-layer attention, router).
- **The memory win is per-cell KV cost, not parameter count:** half the layers × half the global-layer KV heads ≈ ¼ the per-cell cost, which is how 2.5× more context fits in half the KV memory. A categorical difference in deployment class, not an increment.
- **The ~6% MoE PCIe decode penalty** where dense showed none: fixed per-transfer overhead (latency, launch, sync — not bytes) takes a visible bite out of a ~10 ms/token budget that it couldn't take from dense's ~48 ms. Accelerating compute exposes communication costs that were previously hidden.
- **MoE prefill discount survives large batches** — each token routes to 8 of 128 experts regardless of batch size, so prefill compute stays `tokens × active_experts`, contrary to the initial intuition that big prefill batches would light up the whole expert bank.

A near-miss worth recording: the MoE server initially came up on GPUs 0+1 because `CUDA_VISIBLE_DEVICES` was stale from a prior session — caught on the first `nvidia-smi` check *before* the sweep ran. Empirical placement verification, not launcher intent, is the habit this seeded.

## vLLM TP=2 bring-up: six failures, then AWQ-INT4 (Day 4)

Six distinct failure modes preceded the working configuration, each a different layer of the stack:

1. **Pip resolver wall** — pip backtracked to a pre-Gemma-4 vLLM nightly to satisfy `transformers<5`; the official recipe assumes `uv`'s `--index-strategy unsafe-best-match`. Pivot: the team's `vllm/vllm-openai:gemma4` Docker image.
2. **Block-FP8 checkpoint shape mismatch** — the community FP8 checkpoint requires dimensions divisible by 128; Gemma 4's 2112 intermediate dim isn't (and would fail at TP=1 too).
3. **FP8 KV cache needs SM 8.9+** — Inductor's fused KV-write kernel needs a hardware `bf16 → fp8e4nv` cast that Ampere doesn't have.
4. **Marlin FP8 MoE shape-table miss** — expert FFN dim 704 shards to K=352 under TP=2; no tuned config exists, sentinel error.
5. **Triton FP8 MoE rejects vLLM's per-tensor scaling scheme** — handles the shape, not the quant scheme.
6. **`batched_triton` is not CLI-exposed**; every remaining backend is AMD- or Hopper-only. End of the FP8 road on Ampere for this model.

**AWQ-INT4 (`compressed-tensors` path, INT4 Marlin kernels) loaded cleanly** and served the full 262,144-token window at TP=2: 9.09 GiB weights + 10.93 GiB KV per GPU. Engine-reported max concurrency: **24.24× at MML 16,384; 3.91× at MML 262,144** — capacity math from the KV budget, not load tests. The same K=352 shape lacks a tuned INT4 MoE config too, but there it falls back to a slow default and *runs*; single-request decode came out at roughly 10–14 tok/s, well under llama.cpp's 94–112 tok/s on the same model, with the untuned kernel the leading (unverified) explanation.

The cleanest insight of the day: **FP8 weight quantization and FP8 KV cache have different hardware requirements.** Marlin emulates FP8 weights anywhere by upconverting at matmul time; FP8 KV cache needs a chip-level instruction that arrived with Ada/Hopper. They are routinely discussed as one feature; they aren't.

Two framework-behavior contrasts landed as durable facts: vLLM **does not auto-fit** `max-model-len` (it refuses to start; llama.cpp shrinks), and PagedAttention's KV budget is MML-independent (raising MML raises the per-request ceiling, pre-allocating nothing). The observed 24.24/3.91 = 6.2× concurrency ratio against the naive 16× context ratio is the hybrid-attention SWA cap saving ~2.6× at 256K — interpretation from the engine's own math.

## The statmon-ai decision (Day 3) — recorded as a decision, not a measurement

The original plan — fine-tune the 31B Dense to eliminate its ~5K-token system prompt — was abandoned on two grounds: (1) the efficiency argument collapsed, since MoE prefill (3.6× faster) plus llama.cpp's default prefix caching reduce the static prompt's amortized per-request cost to near zero; (2) a prompt audit found ~75% reference documentation + ~10% tool descriptions — exactly the content fine-tuning handles worst (hallucinated enum values, drift against the live CLI) and in-context prompting handles verbatim. Revised approach: deploy the MoE with the full prompt and prefix caching; decouple the fine-tuning learning goal onto a future small-model (3B–7B) exercise where behavioral baking actually applies. Left explicitly open: at ~15–20K prompt tokens, prefix caching alone stops sufficing and retrieval/router/RAG architectures become relevant.

## Prediction misses (preserved)

1. **NVLink-vs-PCIe dense prefill:** predicted ~1.5% penalty from bandwidth×payload math; measured 21–30%. The miss was mechanistic — host-staging fallback (no P2P) plus per-transfer fixed overhead across ~14 chunked transfers, none of which naive bandwidth math contains.
2. **MoE PCIe prefill:** predicted 30–45% (transfer-to-compute-ratio argument); measured 24%. Directionally right, magnitude overshot — transfer/compute overlap means absolute transfer bytes dominate more than the ratio argument suggested. First-order ratio arguments are real effects but not proportional ones.

## Not measured this week

- **Concurrent load, on anything.** The 24.24×/3.91× figures are vLLM's startup capacity computation; llama.cpp multi-user behavior was never exercised. The Day 3 LinkedIn article's multi-user framing was deliberately hedged for this reason.
- **vLLM-vs-llama.cpp like-for-like throughput** on the 26B-A4B — both configurations existed by Day 4 close, but the comparison (and the sweep-script generalization it needs) was deferred to Week 9.

## Methodology lessons logged this week

- **Prefix-sharing synthetic prompts + server-side prefix caching = silently corrupted benchmarks.** Nonce every prompt; verify with the server's own work counters (`timings.prompt_n` vs `usage.prompt_tokens`).
- **OpenAI-compatible endpoints may silently drop native extension fields** — debug output is the only proof an option is honored.
- **Verify GPU placement empirically before measuring** — the stale-`CUDA_VISIBLE_DEVICES` slip would have produced "NVLink" numbers that were PCIe numbers.
- **Predict before measuring, even when the prediction misses** — both misses forced explicit mechanisms that made the corrections legible.
- **P2P capability is motherboard topology, not GPU spec** — two boards with identical slot counts can differ categorically; check `nvidia-smi topo -p2p r` first.
- **"Deployment category" beats raw throughput as an evaluation frame** — dense-at-104K and MoE-at-262K-with-headroom are different products on the same silicon.

## Artifacts produced this week

- `exp2_throughput_sweep.py` — nonce defeat, `usage.prompt_tokens` reporting, `--model-name` parameterization (the seed of the multi-model tool rule)
- `results/exp2_throughput_sweep*.json` — dense NVLink + PCIe sweeps, MoE sweeps
- `~/work/llama.cpp/models/templates/gemma4-no-think.jinja` — thinking-mode-off chat template
- llama.cpp segfault bug report (fixed upstream overnight)
- Two LinkedIn articles: 31B-on-2×3090 day-1 report (`docs/linkedin/2026-04-04-gemma4-31b-rtx3090.md`), dense-vs-MoE head-to-head
- Per-day journals (Days 1–4)

## Carried into Week 9

1. **Framework comparison** — same model, same prompts, both backends; requires generalizing the sweep script to any OpenAI-compatible endpoint with model-name parameterization.
2. **Concurrent-load measurement** — turn the hedged multi-user framing into measured claims.
3. **Untuned INT4 MoE kernel config** (`E=128, N=352` on RTX 3090) — tune and consider contributing upstream; **Marlin FP8 MoE K=352 sentinel error** worth filing as a repro.
4. Triton Inference Server curriculum, displaced from this week.
