# Week 8 Journal — Day 1: Gemma 4 31B Dense on 2x RTX 3090 NVLink

**Training Program:** RTX 3090 AI Infrastructure — Phase 2: Production Inference at Scale
**Date:** April 3, 2026
**Hardware:** 4x RTX 3090 (48GB NVLink pair: GPU 0+2), Ubuntu 24.04, CUDA 12.6
**Model:** Gemma 4 31B Dense IT (released April 2, 2026 — less than 24 hours old)

---

## Objective

Deploy Google's Gemma 4 31B Dense model in INT8 on the NVLink-bridged RTX 3090 pair (GPU 0 + GPU 2) as a locally hosted replacement for the Claude API in the statmon-ai project. This is the first day of a pivot from the original Week 8 curriculum (Triton deep dive) to real-world deployment of a brand-new model.

---

## What Happened

### llama.cpp: Build, Deploy, Hit a Wall

Built llama.cpp from source with CUDA support and downloaded the Q8_0 GGUF (32.6 GB) from `ggml-org/gemma-4-31B-it-GGUF`. The model loaded successfully across both GPUs via layer splitting with `CUDA_VISIBLE_DEVICES=0,2` and `-ngl 999`.

**Startup revealed fascinating architecture details.** Gemma 4 uses a hybrid attention system: 50 sliding-window layers (1024-token window) and 10 global full-context layers. The KV head count varies per layer — most layers have 16 KV heads, but every 6th layer (the global layers) drops to 4. This is a deliberate design: global layers are the most expensive for KV cache since they store the full context, so fewer heads there saves the most memory.

**Memory allocation was tight but functional:**
- GPU 0: 15,326 MiB model weights + ~7.5 GiB KV cache = 22,818 MiB total
- GPU 2: 15,783 MiB model weights + ~7.1 GiB KV cache = 22,910 MiB total
- CPU: 1,428 MiB (token embedding table, memory-mapped)
- Context automatically reduced from 262K to 104,704 tokens to fit

**Thinking mode was on by default**, causing the model to burn all output tokens on internal reasoning (`reasoning_content`) before producing any visible response (`content`). Created a modified Jinja template (`gemma4-no-think.jinja`) that strips the `<|think|>` token from the system prompt and always emits an empty thought block. Result: 3.6x faster response times (2.3s vs 8.3s) with identical generation speed (~24 tok/s).

**Then the crash.** The model segfaults when processing prompts above ~5,400 tokens. Binary search narrowed the boundary precisely: 5,482 tokens works, ~5,600 crashes. The crash occurs after the request completes — the server logs a 200, then segfaults. Reproduced on both single-GPU and multi-GPU configurations, ruling out pipeline parallelism as the cause. This is a core Gemma 4 implementation bug in llama.cpp, filed less than 24 hours after model release.

**This is a blocker for statmon-ai**, which has a ~6K token system prompt before any conversation begins. Median conversations run ~10K tokens.

### Throughput Data (Pre-Crash Range)

| Prompt Tokens | Prefill (tok/s) | Decode (tok/s) | Prefill Time | Wall Time |
|--------------|----------------|----------------|-------------|-----------|
| 488 | 902.5 | 23.8 | 0.54s | 3.2s |
| 922 | 825.9 | 23.6 | 1.12s | 3.9s |
| 1,804 | 901.7 | 23.0 | 2.00s | 4.6s |
| 1,769 | 556.1 | 22.1 | 3.18s | 5.9s |

Decode speed is rock steady at ~23-24 tok/s regardless of prompt size. Prefill runs at ~800-900 tok/s.

### vLLM FP8: Close But Not Enough

Pivoted to vLLM with dynamic FP8 weight-only quantization (`--quantization fp8`), which uses Marlin kernels for FP8 W8A16 on Ampere GPUs. vLLM loaded the model successfully — 16.47 GiB per GPU after FP8 compression — but OOM'd during sampler warmup. The model consumed 70% of each GPU just for weights, leaving almost nothing for KV cache and vLLM's runtime overhead.

Attempted fixes:
- Reduced `max-model-len` from 32K to 16K → still OOM during KV cache allocation
- Reduced `max-num-seqs` to 4, increased `gpu-memory-utilization` to 0.95 → still OOM
- Added `--kv-cache-dtype fp8_e5m2` → incompatible with FP8 weight quantization ("fp8_e5m2 kv-cache is not supported with fp8 checkpoints")
- Added `--enforce-eager` to skip CUDA graphs → different error, same fundamental problem

**Root cause:** vLLM's FP8 weight-only quantization through Marlin produces ~16.5 GiB per GPU. This is larger than llama.cpp's Q8_0 split (~15.5 GiB per GPU) because vLLM doesn't memory-map the embedding table to CPU, and has higher fixed overhead for NCCL communication buffers, attention backends, and sampler state. On 24GB GPUs, that extra ~1 GiB makes the difference between "barely fits" and "doesn't fit."

---

## Key Learnings

### 1. Day-1 Model Deployments Are Rough

Both llama.cpp and vLLM had Gemma 4 support added within hours of release. Both have issues. This is the reality of deploying bleeding-edge models: the tooling needs time to mature. The llama.cpp segfault and vLLM's memory pressure are both things that will be fixed within days or weeks — but aren't usable today for production workloads.

### 2. Weights and KV Cache Are Quantized Independently

Model weights (static, loaded from disk) and KV cache (computed dynamically during inference) are completely separate. You can have INT8 weights with FP16 KV cache, or vice versa. The KV cache is generated at runtime by multiplying input activations against the K/V projection matrices — the resulting vectors are stored in whatever precision the serving framework uses for the cache, not the weight precision. llama.cpp confirmed this explicitly: `K (f16): 4090.00 MiB, V (f16): 4090.00 MiB`.

### 3. Gemma 4's Hybrid Attention Is Memory-Efficient by Design

The sliding-window/global attention split means the KV cache has two components: a large global cache (104K tokens × 10 layers) and a compact SWA cache (1024 tokens × 50 layers). This is far more memory-efficient than a model where all layers maintain full-context KV cache. The variable KV head count (16 for SWA layers, 4 for global layers) further reduces memory by using fewer heads on the layers that store the most tokens.

### 4. Thinking Mode Has Massive Performance Implications

The no-think template reduced response time from 8.3s to 2.3s — a 3.6x improvement — not by changing inference speed, but by eliminating ~200 wasted tokens of internal reasoning. For API-replacement use cases where you want deterministic, fast responses, disabling thinking is essential. The model's generation speed is identical either way; the savings come entirely from not generating unnecessary tokens.

### 5. llama.cpp vs vLLM Memory Overhead

For the same model, llama.cpp uses significantly less GPU memory than vLLM. llama.cpp memory-maps the embedding table to CPU (saving ~1.4 GiB per GPU), has no NCCL buffers, no CUDA graph pools, and no sampler warmup allocation. vLLM's overhead is the price of continuous batching, PagedAttention, and production-grade serving — features that don't matter for single-user deployments but are essential at scale.

### 6. Context Window vs Sequence Length

The context window (256K for Gemma 4 31B) is an architectural ceiling set by the model's positional encoding. The serving framework's sequence length (`max-model-len` in vLLM, `-c` in llama.cpp) is an operational limit that determines KV cache allocation. You choose where to operate within the model's ceiling based on available VRAM and workload requirements. llama.cpp auto-fitted to 104K; vLLM couldn't even fit 8K.

---

## Files Created

```
phase-2-production/week-08/
├── exp2_throughput_sweep.py          # Throughput benchmark script
├── results/
│   └── exp2_throughput_sweep.json    # Partial results (pre-crash range)
└── (bug report drafted for llama.cpp GitHub)
```

Also created:
- `~/work/llama.cpp/models/templates/gemma4-no-think.jinja` — Modified chat template with thinking disabled

---

## Current State

- **llama.cpp Q8_0:** Functional up to ~5,400 prompt tokens. Segfaults above that. Bug report prepared.
- **vLLM FP8:** Cannot start — OOM during initialization. Needs either a true INT8 W8A8 compressed-tensors checkpoint or more aggressive quantization.
- **statmon-ai integration:** Blocked by the 5.4K token limit (system prompt alone exceeds it).

---

## Next Steps

- **Tomorrow:** Deploy Gemma 4 26B MoE via llama.cpp. Fits on a single 3090 at Q8_0 (~18GB). Only 3.8B active parameters — reportedly 97% of dense 31B quality. Run throughput benchmarks and functional tests.
- **This week:** File the llama.cpp bug report. Monitor for fixes and community INT8 checkpoints.
- **LinkedIn post angle:** Honest day-1 deployment report — what works, what doesn't, practical guidance for consumer GPU users. The 26B MoE "what actually works today" angle may be more valuable than the 31B dense story.
- **When llama.cpp is fixed:** Resume 31B dense benchmarks across full context range, including KV cache quantization experiments to push toward 256K context.

---

*Hardware: 4x RTX 3090 (96GB total), NVLink bridge GPU 0↔GPU 2*
*Frameworks: llama.cpp (b8660), vLLM 0.19.0*
*Model: google/gemma-4-31B-it (BF16 base), ggml-org/gemma-4-31B-it-GGUF (Q8_0)*
