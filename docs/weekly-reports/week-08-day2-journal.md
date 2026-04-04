# Week 8 Journal — Day 2: Gemma 4 31B — Bug Fixed, Clean Throughput Curve, and a Consumer Topology Lesson

**Training Program:** RTX 3090 AI Infrastructure — Phase 2: Production Inference at Scale
**Date:** April 4, 2026
**Hardware:** 4x RTX 3090 (48GB NVLink pair: GPU 0+2), Ubuntu 24.04, CUDA 13.0
**Model:** Gemma 4 31B Dense IT (Q8_0 GGUF via llama.cpp)

---

## Objective

Continue yesterday's Gemma 4 31B deployment work. Verify the overnight fix for the prompt-length segfault, complete the throughput sweep across the full context range, quantify NVLink's contribution to pipeline parallelism versus a PCIe fallback path, and capture the findings in a publishable write-up.

---

## What Happened

### Bug Fix Confirmed

The llama.cpp maintainers turned around the segfault fix overnight. Pulled latest, rebuilt, and restarted `llama-server` with the original configuration — no `--cache-ram 0` workaround, just the layer-splitting command used on Day 1 before the crash was discovered. The throughput sweep ran end-to-end without any crashes, including requests well past the old 5,500-token boundary. An impressive response for a day-1 model support bug in an open-source project.

### Initial Throughput Sweep — and an Anomaly

Ran `exp2_throughput_sweep.py` against the NVLink pair (GPU 0+2) across seven target prompt sizes from 500 to 32,000 tokens. Got clean-looking results with one exception: the row for target=4,000 tokens reported a prefill rate of 547 tok/s — roughly 40% slower than the ~900 tok/s steady state of neighboring rows. First hypothesis was thermal throttling, background process, or simple variance. Reran the sweep and got 547 tok/s again, essentially exact reproduction. Something structural was going on, not noise.

### The Cache-Reuse Investigation

This became the methodology lesson of the day and took substantially more time than the sweep itself.

Looking at the sweep output more carefully, I noticed the "actual prompt tokens" field was landing at suspiciously clean ratios relative to targets — exactly 44% of target from the target=4K row onward. Too clean to be a tokenizer efficiency effect; that kind of precise scaling is almost always an arithmetic artifact.

Added a debug print to surface both `timings.prompt_n` and `usage.prompt_tokens` from the llama.cpp response. The reveal was immediate: from the target=4K row onward, these two fields diverged by exactly 2:1. The smaller number (what the script had been reporting as "actual prompt tokens") was the number of tokens actually processed by prefill on that request. The larger number was the true prompt size.

**Root cause: llama-server's slot cache reuse.** Because the benchmark's `build_prompt()` function uses the same framing text and filler sentence for every row, consecutive prompts share a long common prefix. When target=4K runs after target=2K, the server's slot cache already contains the first ~1,800 tokens, so it only needs to prefill the new ~1,800 tokens at the end. The `timings.prompt_n` field reports that number (new work), not the full prompt size. Meanwhile, `timings.prompt_ms` measures the time to process just those new tokens, so the computed prefill rate was artificially *low* at the 4K row — a fixed per-request overhead (cache scanning, prefix matching, splicing) was being amortized over only half as many tokens as expected. The larger rows (7K, 14K, 28K) were also reusing cache, but the fixed overhead was diluted across enough new tokens that the rate looked fine.

Two fixes were needed:

1. Report `usage.prompt_tokens` (true prompt size) in the table column — the field the script had been using was a work counter, not a size.
2. Defeat cache reuse itself so the benchmark measures cold prefill consistently.

First attempt at #2: added `"cache_prompt": false` to the request payload. Reran, and the debug output showed the two fields still diverging. Turned out llama-server's OpenAI-compatible endpoint silently drops the `cache_prompt` field — it's a native llama.cpp extension that only works on the `/completion` endpoint, not `/v1/chat/completions`. The server accepts the field and discards it with no error or warning.

Working fix: prepend a UUID nonce to each prompt. Cache reuse operates on longest-common-prefix matching, so if the very first token differs between requests, the common prefix is zero and the server has no choice but to cold-prefill everything. Added `uuid.uuid4().hex` to the prompt template. Reran. Debug output now showed `timings.prompt_n == usage.prompt_tokens` for every row, and the prefill curve emerged as monotonically well-behaved.

### Clean Throughput Results

| Prompt Tokens | Prefill (tok/s) | Decode (tok/s) | Prefill Time | Wall Time |
|---------------|-----------------|-----------------|---------------|-----------|
| 519 | 862.1 | 23.9 | 0.60s | 3.34s |
| 954 | 843.7 | 23.6 | 1.13s | 4.26s |
| 1,837 | 919.7 | 23.0 | 2.00s | 5.18s |
| 3,585 | 946.9 | 22.1 | 3.79s | 7.84s |
| 7,086 | 1,088.6 | 21.6 | 6.51s | 12.10s |
| 14,085 | 1,173.3 | 21.3 | 12.00s | 18.21s |
| 28,085 | 1,157.9 | 20.4 | 24.25s | 32.73s |

**Decode:** 23.9 tok/s at 519 tokens → 20.4 tok/s at 28,085 tokens. A 14.6% drop across a 54x span in context length. Decode is firmly weight-bandwidth-bound at every point on the curve; the KV cache reads never grow large enough to tip it into a KV-bound regime.

**Prefill:** Rises from ~862 tok/s at small prompts to a plateau of ~1,170 tok/s from 14K onward. The plateau is ~18% higher than the original artifact-laden measurement suggested — a meaningful correction for TTFT planning. The slight drop from 14K (1,173 tok/s) to 28K (1,158 tok/s) is the first hint of the O(n²) attention regime from Gemma 4's global-attention layers starting to contribute measurably at larger sequence lengths. At 1.3% it's nothing to worry about, but it indicates the shape of the curve beyond 28K.

### NVLink vs PCIe Comparison

With a clean baseline in hand, the obvious follow-up was to quantify NVLink's actual contribution to pipeline-parallel inference on this hardware. Flipped `CUDA_VISIBLE_DEVICES=0,2` (NVLink pair) to `0,1` (no NVLink, PCIe-only path) and reran the sweep.

My prediction was that the impact would be small. Pipeline parallelism only transfers activations once per forward pass, and the payload is small: ~10KB per token during decode, ~287MB for a 28K prefill pass. Naive bandwidth math suggested ~1.5% hit at 28K.

The actual result: decode was essentially unaffected as predicted (23.4 tok/s vs 23.9 tok/s at small prompts, both curves flat), but **prefill took a 21% hit at 28K and peaked at 30% in the mid-range**. An order of magnitude larger than my prediction.

### The Topology Finding

Ran `nvidia-smi topo -p2p r` to diagnose. The peer-to-peer matrix showed `CNS` (chipset not supported) for every non-NVLink pair in the system — only the NVLink-bridged GPU 0+2 pair had `OK`. So when the benchmark ran with `CUDA_VISIBLE_DEVICES=0,1`, CUDA had no direct peer path available and silently fell back to staging through pinned host memory: GPU 0 → system RAM → GPU 1. That's a two-hop path rather than direct DMA, and on top of the raw bandwidth cost, it adds CUDA stream synchronization and host-side memory operations for every transfer. llama.cpp also chunks prefill into batches (default ~2048 tokens in recent builds), so a 28K prompt isn't one handoff — it's roughly 14 separate transfers, each paying its own fixed per-transfer overhead.

This is topology-specific to the motherboard. GPUs 0 and 2 share access through the NVLink bridge that routes around the normal PCIe hierarchy, while slots for GPUs 1 and 3 hang off the chipset rather than the CPU's own PCIe root complex. Without direct CPU-attached paths to every slot, CUDA can't enable peer access between non-NVLink pairs. This is a common layout on consumer motherboards but rarely documented — two boards with identical spec sheets (same slot count, same nominal lane widths) can behave very differently depending on which slots share a root complex with the CPU.

The finding from Week 7 ("NVLink is a precondition for tensor parallelism on this hardware") takes on a sharper meaning in light of this: it wasn't just about NVLink's higher bandwidth, it was about peer access existing at all. Without the NVLink bridge, even a hypothetical 4-way TP setup on this motherboard would have been paying the host-staging penalty on every cross-GPU communication.

### LinkedIn Article Drafted

Captured the experiment narrative as a long-form LinkedIn article draft. Focus: what it takes to run Gemma 4 31B on 2x RTX 3090s today (hybrid attention architecture, Q8_0 quantization, llama.cpp layer splitting, thinking-mode patch), the day-1 bug story, the NVLink prefill/decode characterization, and — as the broader lesson — the topology-dependent PCIe fallback finding with `nvidia-smi topo -p2p r` as a pre-purchase diagnostic. Approximately 850 words, saved to `docs/linkedin/` for publication.

---

## Key Learnings

1. **Prefix-sharing synthetic prompts interact badly with server-side cache reuse.** Benchmarks built by concatenating common framing with variable filler text are a natural target for slot cache optimization in production inference servers. The server "helpfully" reuses the common prefix and only processes new tokens — but the reported throughput is computed against the reduced work, so the numbers look systematically off for the wrong reason. The clean fix is a unique nonce per request, which defeats prefix matching at the content level and forces cold prefill.

2. **`timings.prompt_n` and `usage.prompt_tokens` mean different things in llama.cpp responses.** `prompt_n` is a work counter — tokens that actually went through prefill this request. `usage.prompt_tokens` is the total prompt size. For benchmark reporting, `usage.prompt_tokens` is the correct column; the other field is useful only when explicitly trying to measure the cost of *new* work in a cache-reuse scenario.

3. **llama.cpp's OpenAI-compatible endpoint silently drops unknown fields.** The `cache_prompt` extension works on the native `/completion` endpoint but not on `/v1/chat/completions`. No error, no warning — the server accepts the field and discards it. Feature parity across endpoints is not guaranteed, and debug output is the only reliable way to confirm that an extension is actually being honored.

4. **NVLink's value for pipeline parallelism is modest compared to tensor parallelism.** Week 7 saw 9.5x improvement from NVLink under TP=2 because tensor parallelism does an all-reduce after every layer. Pipeline parallelism does only one activation handoff per forward pass, so the raw bandwidth requirement is two orders of magnitude smaller. The measured prefill improvement is 21% at 28K — meaningful but not transformative, and vastly smaller than the TP case. Different parallelism strategies have very different bandwidth profiles, and the "how much does NVLink matter?" answer depends entirely on which strategy is in use.

5. **Peer-to-peer DMA between consumer GPUs is not guaranteed by topology alone.** On motherboards where additional GPU slots hang off the chipset rather than the CPU's own PCIe root complex, CUDA denies peer access and silently falls back to staging through pinned host memory. This doubles the effective PCIe traffic, serializes with host memory operations, and adds stream synchronization on every transfer. The result can be substantially worse than the link's nominal bandwidth would suggest. The naive "bandwidth × payload = cost" calculation missed by an order of magnitude precisely because it didn't account for host staging. `nvidia-smi topo -p2p r` is the diagnostic that would have flagged this before the benchmark ran.

6. **The true prefill ceiling for Gemma 4 31B Q8_0 on NVLink is ~1,170 tok/s, not ~990.** The original sweep measurement underreported the plateau by ~18% because fixed per-request cache-reuse overhead was folded into the rate calculation on reused rows. This matters for capacity planning, TTFT estimation, and any comparison against other frameworks or models.

---

## Files Created / Modified

```
phase-2-production/week-08/
├── exp2_throughput_sweep.py              # Updated: UUID nonce in build_prompt(),
│                                         # usage.prompt_tokens in reporting path,
│                                         # debug print for timings.prompt_n vs usage
├── results/
│   ├── exp2_throughput_sweep_nvlink.json # Clean NVLink sweep (GPU 0+2)
│   └── exp2_throughput_sweep_pcie.json   # PCIe fallback sweep (GPU 0+1)

docs/linkedin/
├── README.md                             # Publication index for LinkedIn directory
└── 2026-04-04-gemma4-31b-rtx3090.md      # LinkedIn article draft
```

---

## Current State

- Gemma 4 31B Q8_0 runs reliably on the 2x RTX 3090 NVLink pair across the full 104K context window llama.cpp allocates for it.
- Throughput is cleanly characterized: ~1,170 tok/s prefill plateau, ~20–24 tok/s decode. The curve is well-behaved with no artifacts remaining in the data.
- NVLink contribution to pipeline parallelism is quantified at ~21% prefill improvement at 28K tokens — far smaller than the 9.5x from tensor parallelism in Week 7, as expected given the different communication patterns.
- Consumer motherboard P2P topology is documented with concrete numbers and a diagnostic command.
- LinkedIn article draft is ready for publication.

---

## Next Steps

- Publish the LinkedIn article.
- Decide whether to continue the Gemma 4 deployment investigation for another day (possibilities include concurrent-request testing, KV cache quantization experiments to push beyond 104K context, or a direct comparison against the 26B MoE variant discussed on Day 1) or return to the originally planned Week 8 curriculum (Triton deep dive, NVLink TP=2 vs 4-way data parallel).
- Consider filing a documentation issue with llama.cpp about `cache_prompt` being silently ignored on the `/v1/chat/completions` endpoint — the silent-drop behavior cost several hours of debugging today and would likely trip up anyone else trying to defeat cache reuse the "obvious" way.

---

*Hardware: 4x RTX 3090 (96GB total), NVLink bridge GPU 0↔GPU 2*
*Framework: llama.cpp (post-fix build, Gemma 4 segfault resolved)*
*Model: ggml-org/gemma-4-31B-it-GGUF Q8_0*
