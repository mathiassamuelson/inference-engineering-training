# Week 8, Day 2: Gemma 4 26B A4B on vLLM TP=2 over NVLink

**Date:** April 7, 2026
**Hardware:** 4x RTX 3090 (96 GB total), Gigabyte B650 Eagle AX, Ubuntu 24.04, CUDA 12.6
**Topology:** NVLink bridge GPU 0 ↔ GPU 2 (NV4, ~100 GB/s bidirectional)
**Framework:** vLLM 0.18.2rc1.dev73+gdb7a17ecc (from `vllm/vllm-openai:gemma4` Docker image)
**Models attempted:** `protoLabsAI/gemma-4-26B-A4B-it-FP8`, `google/gemma-4-26B-A4B-it` (BF16 base), `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`
**Model that worked:** `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`

---

## Executive Summary

Day 2 set out to close Week 8 by deploying Gemma 4 26B A4B (the MoE variant, 25.2B total / 4B active) on vLLM with tensor parallelism across the NVLink pair. The session walked through **six distinct failure modes** before landing on a working configuration. The headline result is significant: **vLLM holds the full 262,144-token context window with 3.91x concurrency on this hardware**, a configuration llama.cpp could not fit yesterday on the 31B Dense (which auto-shrunk to 104,704 tokens on the same two cards). At a more practical 16,384-token `max-model-len`, the same configuration delivers **24.24x concurrency**, comfortably covering the statmon-ai operational range with substantial headroom.

The path to that result is the more important content. Each failure was a real, learnable thing about Day-1 deployment of a brand-new MoE architecture on consumer Ampere hardware. The chain ran: pip resolver wall → pre-quantized FP8 block-shape mismatch → FP8 KV cache hardware incompatibility → Marlin FP8 MoE shape table miss → Triton FP8 MoE quant scheme mismatch → end of user-selectable FP8 backends → AWQ-INT4 pivot success. The cumulative signal is that vLLM serving Gemma 4 26B A4B at TP=2 on consumer Ampere is in genuinely uncharted territory as of today, and the FP8 path simply isn't viable yet on this combination. AWQ-INT4 sidesteps the entire FP8 code path through `compressed-tensors` and works cleanly.

---

## The Working Configuration

```bash
docker run --rm -it \
  --gpus '"device=0,2"' \
  --ipc=host \
  --shm-size 16G \
  --network host \
  -e HF_TOKEN=$HF_TOKEN \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:gemma4 \
  cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":0,"audio":0}' \
  --host 0.0.0.0 \
  --port 8000
```

Key startup observations from the working run:

| Metric | Value |
|---|---|
| Model loading memory (per GPU) | 9.09 GiB |
| Available KV cache memory (per GPU) | 10.93 GiB |
| GPU KV cache size (token-blocks) | 95,472 tokens |
| Max concurrency at `max-model-len 16384` | **24.24x** |
| Max concurrency at `max-model-len 262144` | **3.91x** |
| Attention backend selected | TRITON_ATTN (forced — heterogeneous head dimensions) |
| Linear quant kernel | MarlinLinearKernel (CompressedTensorsWNA16) |
| MoE quant method | CompressedTensorsWNA16MoEMethod |
| MoE kernel config status | **Default fallback (untuned)** — see Open Questions |
| Engine init time | 46–48 seconds |

---

## The Failure Chain

### Failure 1: Pip dependency resolver wall

**Attempt:** Create a side venv `~/ai-inference-gemma4` and install vLLM nightly + `transformers==5.5.0` per the official Gemma 4 recipe.

**Failure:** Pip pulled `vllm 0.19.1rc1.dev86+g70406eb1d` — a nightly that **predates the Gemma 4 merge**. After installing transformers 5.5.0 separately, pip's resolver flagged the conflict:

```
vllm 0.19.1rc1.dev86 requires transformers<5,>=4.56.0,
but you have transformers 5.5.0 which is incompatible.
```

**Root cause:** Pip's standard resolver respects every constraint in the dependency tree. When it saw `compressed-tensors` requiring `transformers<5.0.0`, it walked backwards through vLLM versions until it found one whose own `transformers<5` ceiling matched. That version (the dev86 nightly) is from before the Gemma 4 PRs (#38826, #38847) merged. The official recipe is written for `uv`, which has `--index-strategy unsafe-best-match` to bypass exactly this kind of constraint wall — pip lacks the equivalent flag.

**Pivot:** Abandoned the bare-metal venv, pulled the prebuilt `vllm/vllm-openai:gemma4` Docker image (23.9 GB) which the vLLM team explicitly recommends "for out-of-box usage."

**Lesson:** When upstream provides a pinned-build Docker image with a "we recommend this" label, the warning isn't decorative. Day-0 model launches against fast-moving frameworks have dependency graphs that are easier to ship as containers than to reproduce in venv.

### Failure 2: protoLabsAI FP8 block-shape mismatch

**Attempt:** Launch vLLM in the gemma4 container against `protoLabsAI/gemma-4-26B-A4B-it-FP8` (a community pre-quantized FP8 checkpoint, ~25 GB on disk).

**Failure:** During model construction, before any forward pass:

```
ValueError: Weight input_size_per_partition = 1056 is not divisible
by weight quantization block_k = 128.
```

**Root cause:** The protoLabs checkpoint uses a **block-FP8 quantization scheme** that requires every tensor's input dimension to be divisible by `block_k=128`. Gemma 4 26B A4B's shared MLP has an intermediate dimension of 2112, which isn't divisible by 128 (2112 / 128 = 16.5). Sharded across TP=2 it becomes 1056, still not divisible. The hard validator in `validate_fp8_block_shape` correctly refused to proceed. **This checkpoint would have failed at TP=1 too** — it's not a topology issue, it's a structural incompatibility between the chosen quantization scheme and the model's hidden dimensions.

**Pivot:** Switched to canonical BF16 weights (`google/gemma-4-26B-A4B-it`, ~50 GB) plus vLLM's on-the-fly FP8 quantization (`--quantization fp8`), which uses per-tensor scaling rather than block-128 and has no divisibility constraint.

**Lesson:** Pre-quantized FP8 checkpoints carry block-shape constraints that interact non-obviously with the underlying model's hidden dimensions. Community quantizers building checkpoints for new architectures are likely to ship configurations that haven't been exercised against every model's actual weight shapes.

### Failure 3: FP8 KV cache requires hardware that Ampere doesn't have

**Attempt:** Launch with the BF16 base model + `--quantization fp8 --kv-cache-dtype fp8`.

**Failure:** Past model loading, past torch.compile, **inside Inductor's Triton codegen** for a fused KV cache write kernel:

```
ValueError: type fp8e4nv not supported in this architecture.
The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')
```

**Root cause:** RTX 3090 is Ampere (SM 8.6). **Native FP8 instructions arrived with Ada/Hopper (SM 8.9+).** vLLM correctly handled this for the **weight quantization** path by selecting `MarlinFP8ScaledMMLinearKernel` — Marlin stores weights as FP8 but upconverts to bf16 at matmul time, no native FP8 compute required. But the **KV cache write path** is different: Inductor generates a Triton kernel that needs a hardware-level `bf16 → fp8e4nv` cast, which is a chip-level instruction Ampere simply doesn't have.

**Pivot:** Drop `--kv-cache-dtype fp8`. Use bf16 KV cache (the default). The 26B A4B's hybrid attention is frugal enough on KV that we don't need FP8 KV to fit reasonable contexts.

**Lesson — the cleanest insight from the day:** **FP8 weight quantization and FP8 KV cache have different hardware requirements.** Weight quant runs anywhere via Marlin emulation; KV cache FP8 needs Ada or newer. They're often discussed together as if they're a single feature, but they're not.

### Failure 4: Marlin FP8 MoE has no tuned config for K=352

**Attempt:** Same command as Failure 3, but with FP8 KV cache disabled.

**Failure:** Past model loading, past torch.compile, **during the profile_run dummy forward pass** (the memory-sizing step before KV cache allocation):

```
RuntimeError: Invalid thread config: thread_m_blocks = 4,
thread_k = -1, thread_n = -1, num_threads = -1
for MKN = [16384, 352, 2816] and num_bits = 8 [...]
```

**Root cause:** Marlin MoE GEMM has a hand-tuned table of supported (M, K, N) shapes; the `-1, -1, -1` sentinel means *"no valid config found in the lookup table for this shape."* The dimensions decode as: M=16384 (dummy batch), N=2816 (model hidden dim), and **K=352**. Where does 352 come from? Gemma 4 26B A4B's expert FFN intermediate dimension is **704**. Sharded across TP=2, each GPU holds half of each expert: 704 / 2 = **352**. This is **structurally the same bug as Failure 2** — Gemma 4's hidden dimensions don't divide cleanly under TP=2 in ways that surface differently in different code paths.

**Pivot:** Override the auto-selected MoE backend with `--moe-backend triton`. The general-purpose Triton MoE kernel doesn't have a tuned-config lookup table at all and can theoretically handle arbitrary K dimensions including 352.

### Failure 5: Triton FP8 MoE doesn't accept vLLM's per-tensor scaling scheme

**Failure:** With `--moe-backend triton`, much earlier in startup — during quant method initialization, before model weights even load:

```
ValueError: FP8 MoE backend TRITON does not support the deployment configuration
since kernel does not support quantization scheme
QuantKey(f8e4m3fn,scale(f32,static,per_tensor),symmetric)
xQuantKey(f8e4m3fn,scale(f32,dynamic,per_tensor),symmetric).
```

**Root cause:** Triton MoE handles arbitrary K but only accepts certain combinations of weight/activation scaling granularities. vLLM's on-the-fly FP8 produces W8A8 with **per-tensor static** weight scaling and **per-tensor dynamic** activation scaling. The Triton MoE kernel apparently expects either block-wise or channel-wise scaling, not per-tensor. So even though Triton can handle the *shape* problem, it rejects the *quant scheme*.

**Pivot attempt:** Try the next candidate from the auto-selection list — `batched_triton`.

### Failure 6: batched_triton is not user-selectable

**Failure:** The CLI rejects the flag value:

```
vllm serve: error: argument --moe-backend: invalid choice: 'batched_triton'
(choose from aiter, auto, cutlass, deep_gemm, flashinfer_cutedsl,
flashinfer_cutlass, flashinfer_trtllm, marlin, triton)
```

**Root cause:** `batched_triton` appears in vLLM's internal MoE backend candidate set but isn't exposed via the CLI. Of the user-selectable options, everything except already-tried `marlin` and `triton` is either AMD-only (`aiter`), Hopper-only (`cutlass`, `deep_gemm`, the flashinfer family), or otherwise inapplicable on RTX 3090. **End of the FP8 backend road on Ampere for this model.**

**Pivot — the big one:** Abandon FP8 entirely and try AWQ-INT4 with `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`. AWQ uses a completely separate quantization code path (`compressed-tensors`), different kernels (INT4 Marlin), different shape support tables, and is one of vLLM's most mature paths on Ampere. It also produces a smaller checkpoint (~13 GB on disk vs ~25 GB FP8 vs ~50 GB BF16).

### Success: AWQ-INT4 via compressed-tensors

The AWQ checkpoint loaded cleanly through the `compressed-tensors` quantization path:

- `Using MarlinLinearKernel for CompressedTensorsWNA16` (linear layers)
- `Using CompressedTensorsWNA16MoEMethod` (MoE layers)
- Both routed to INT4 Marlin kernels with shape support that *does* cover Gemma 4's dimensions

There is one persistent warning in the working configuration that's worth flagging:

```
WARNING [fused_moe.py:1090] Using default MoE config. Performance might be sub-optimal!
Config file not found at .../E=128,N=352,device_name=NVIDIA_GeForce_RTX_3090,dtype=int4_w4a16.json
```

The same K=352 shape that broke Marlin FP8 MoE in Failure 4 *also* lacks a tuned config in the INT4 W4A16 MoE table for this GPU — but here it falls back to a generic default config and **runs** instead of erroring. The cost is throughput: the configuration is correct but not optimized. Any single-request decode rates measured against this server will be conservative estimates compared to what's theoretically achievable.

A sanity request returned coherent text in 70 completion tokens, no truncation, stable decode loop. The single-request decode rate from vLLM's 10-second windowed averages worked out to roughly 10–14 tok/s — meaningfully slower than llama.cpp delivered yesterday on the same model. The untuned MoE kernel is the most likely explanation, and continuous batching at higher concurrency is where the real comparison should happen.

---

## Concurrency at Operational Context Sizes

After confirming the server worked at `max-model-len 16384` (chosen to comfortably cover statmon-ai's 7K–15K range), restarted with `max-model-len 262144` to match yesterday's llama.cpp configuration for the 31B Dense run.

| `max-model-len` | KV cache budget | Max concurrency |
|---:|---:|---:|
| 16,384 | 95,472 tokens | **24.24x** |
| 262,144 | 95,472 tokens | **3.91x** |

Two observations on these numbers:

**KV cache budget is identical between runs** — same 10.93 GiB available per GPU, same 95,472 token-block budget. This is a fundamental architectural difference from llama.cpp: vLLM's PagedAttention allocates blocks dynamically per request, so increasing `max-model-len` doesn't pre-allocate anything, it just raises the per-request ceiling on consumption. llama.cpp commits its cache layout at startup.

**Hybrid attention savings are visible in the concurrency math.** Naive calculation: going from 16K to 256K context per request should reduce concurrency by 16x (24.24 / 16 = 1.52x). The actual ratio is 24.24 / 3.91 = 6.2x. The 2.6x gap between naive prediction and observed reality is the SWA cap doing its job: 25 of the 30 layers only ever store 1024 tokens regardless of conversation length, so only the 5 global layers grow linearly with context. At 256K context, hybrid attention saves a factor of ~2.6x on KV cache compared to a fully-global-attention model of the same size.

**3.91x at 262K context means three full-context concurrent users plus a partial fourth on two RTX 3090s.** Yesterday's 31B Dense run on llama.cpp at the same target `max-model-len 262144` got auto-shrunk to 104,704 tokens because it couldn't fit the full context across the same hardware. **The configuration vLLM is now running was literally not achievable on llama.cpp on this hardware yesterday** — partly because the 26B A4B has half as many layers (30 vs 60), partly because INT4 weights are smaller than Q8_0, and partly because PagedAttention is more memory-efficient than llama.cpp's contiguous allocation. That belongs in the framework decision matrix.

---

## Key Learnings

**FP8 weight quantization and FP8 KV cache have different hardware requirements.** This is the cleanest single insight from the day. Marlin emulates FP8 weights via upconversion at matmul time and runs anywhere; FP8 KV cache needs a hardware-level `bf16 → fp8e4nv` cast that only exists on Ada/Hopper (SM 8.9+). On Ampere you can have one but not the other. They're often discussed as if they're a single feature.

**vLLM does not auto-fit `max-model-len` to available memory.** llama.cpp shrunk 262144 → 104704 yesterday when it couldn't fit. vLLM refuses to start if `max-model-len` doesn't fit — there is no auto-shrink. The good news is that bisecting to find the largest viable value is straightforward; the bad news is you have to do it manually. (For 26B A4B AWQ-INT4 on this hardware, the answer turned out to be "the architectural ceiling fits, no bisect needed.")

**Day-1 deployment of brand-new MoE architectures requires walking through multiple kernel/quant compatibility layers.** Single failures don't tell you the whole picture. Each failure tonight was a different layer of the stack rejecting a different aspect of the configuration: dependency resolution, weight quantization scheme, KV cache hardware support, MoE kernel shape table, MoE kernel quant scheme, CLI surface area. The composition of constraints is what determines whether a deployment is viable, not any single one.

**Community pre-quantized checkpoints carry their own constraints that may not match the model's hidden dimensions.** protoLabs FP8 was unusable not because of any vLLM bug or any RTX 3090 limitation, but because the checkpoint's block-128 quantization scheme is structurally incompatible with Gemma 4's intermediate dimensions. Future-me should treat "is there a community quant for this model?" as the start of the question, not the end — the next question is "and does its quantization scheme play well with the model's actual weight shapes."

**Hybrid attention's KV savings are second-order but real.** At long context, going from full-attention to a 25/5 SWA/global split bought a 2.6x reduction in cache pressure compared to the naive expectation. That's the difference between "1.5x concurrency at 256K" and "3.9x concurrency at 256K." For long-context deployments, this is a significant capacity multiplier.

**The MoE tuned-config gap is a real but sub-optimal-performance issue, not a correctness issue.** The fallback default config runs and produces correct output. It just isn't fast. Any throughput numbers measured tonight should carry an asterisk noting "with an untuned MoE kernel." If the framework comparison work in Week 9 wants a fair vLLM-vs-llama.cpp number for this model, the tuned config gap is the biggest variable to be aware of.

---

## Open Questions and Future Work

**Throughput sweep against the AWQ configuration** is the natural Week 9 starting point. The existing throughput sweep script from the llama.cpp work will need to be generalized to accept any OpenAI-compatible endpoint and parameterize on model name (per the multi-model tool design rule). The eventual deliverable is a like-for-like comparison: same script, same prompts, same concurrency levels, two backends.

**The untuned MoE config gap.** The missing file is `E=128,N=352,device_name=NVIDIA_GeForce_RTX_3090,dtype=int4_w4a16.json`. Generating one would require running vLLM's MoE kernel autotuner against this specific shape on this specific GPU. If the tuned config materially improves throughput, contributing it upstream would be a small but real contribution to vLLM and would benefit anyone else running 26B A4B on RTX 30-series cards.

**The Marlin FP8 MoE shape table gap (Failure 4)** is a bug worth filing with the vLLM team. K=352 is a real shape produced by a real Google-released model under standard TP=2 sharding, and the kernel returns a sentinel error rather than a meaningful "no config available" message. A clean repro would be valuable to upstream.

**Single-request decode rate measurement.** The 10-second windowed averages from vLLM's logger don't give per-request rates directly. A proper measurement (instrumenting the benchmark script to record per-completion timing) is part of the Week 9 throughput script generalization.

**Comparison against the llama.cpp 26B A4B numbers** from earlier this week is the primary outstanding analysis. We now have configurations from both frameworks running the same model at the same context windows. The next question is whether vLLM's continuous batching advantage at concurrency outweighs llama.cpp's apparent advantage in single-request decode (driven at least in part by the untuned MoE kernel on the vLLM side).

---

## Files Created

```
phase-2-production/week-08/
└── (no scripts created today — server-side configuration only)
```

No new scripts went into the repo today; all the work was launch-command iteration and journal documentation.

---

## Status

- **Gemma 4 26B A4B serving on vLLM TP=2 over NVLink:** ✅ Working
- **Configuration:** AWQ-INT4 via `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`, `compressed-tensors` quant path, Marlin INT4 kernels
- **Context windows characterized:** 16,384 (24.24x concurrency) and 262,144 (3.91x concurrency)
- **Throughput characterization:** Deferred to Week 9 (throughput sweep script needs generalization first)
- **FP8 path:** Confirmed not viable on Ampere for this model with current vLLM. Worth re-checking when either (a) vLLM adds the missing Marlin FP8 MoE configs or (b) someone publishes an FP8 checkpoint that doesn't use block-128 weight quantization.

---

*Frameworks: vLLM 0.18.2rc1.dev73 (vllm/vllm-openai:gemma4 image), llama.cpp (b8664)*
*Models tested: cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit (working), protoLabsAI/gemma-4-26B-A4B-it-FP8 (block-shape failure), google/gemma-4-26B-A4B-it (FP8 backend chain failures)*
