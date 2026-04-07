# Gemma 4 MoE vs Dense on Two RTX 3090s: Same Hardware, Same Script, Very Different Models

A few days ago I wrote up deploying @Google DeepMind's Gemma 4 31B Dense on a pair of @NVIDIA RTX 3090s. The natural follow-up was obvious: run the same sweep on the 26B-A4B MoE variant and see how two models from the same family, nominally in the same weight class, actually compare on identical hardware. This turned out to be a more interesting comparison than I expected — not because one model "won," but because the differences show up in places the parameter counts don't hint at.

**The setup was deliberately boring.** Same hardware: two RTX 3090s with an NVLink bridge between GPU 0 and GPU 2 on a Gigabyte B650 motherboard. Same framework: @ggerganov's llama.cpp (post-Gemma-4-fix build) with layer-splitting pipeline parallelism. Same quantization: Q8_0 GGUF from the @Hugging Face GGUF team. Same benchmark script, unchanged except for the model name argument. Same prompt sweep from ~500 to ~28,000 tokens. The only variable was which model llama-server loaded.

**On paper, the models are siblings.** Gemma 4 31B Dense has 30.7B parameters, 60 layers, hidden dim 5376. Gemma 4 26B-A4B MoE has 25.2B total parameters with 4B active per token via 128 experts of which 8 are routed, 30 layers, hidden dim 2816. Structurally, the MoE is almost exactly a half-depth, half-width version of the dense model with MoE FFN blocks swapped in. That symmetry made it unusually easy to reason about where differences would come from.

**Decode speed: a clean 4.7x across the board.** Single-stream decode throughput on NVLink was 112 tok/s at short context and 94 tok/s at 28K on the MoE, against 24 tok/s and 20 tok/s respectively for the dense model. The ratio is flat — 4.6x to 4.7x at every context length measured. This is the ideal shape for a decode-speedup result: it tells you the MoE advantage is purely a weight-bandwidth story that doesn't interact with sequence length. The theoretical ceiling from the active-parameter ratio (4B active vs 30.7B dense) is about 7.7x; the measured 4.7x is below that because decode isn't purely FFN — KV cache traffic, attention compute, and router overhead all happen per token regardless of MoE, and those unchanged costs dilute the advantage. A 61% realized efficiency against the theoretical ceiling is respectable.

**Prefill speed: 3.6x at long context, and the advantage grows.** I expected prefill to shrink the MoE discount — my prior was that at prefill batch sizes, enough tokens would route to enough experts that you'd end up activating most of the expert bank anyway and lose the compute savings. That framing was wrong. What matters for prefill FLOPs is `tokens × active_experts_per_token × expert_size`, not `all_experts × expert_size`. Each token still routes to only 8 of 128 experts regardless of batch size, so the MoE discount applies cleanly. The speedup actually grows with context — 2.6x at 500 tokens, 3.6x at 28K — because Gemma 4's hybrid attention (25 of 30 SWA layers with a 1024-token window, 5 global layers) keeps attention nearly linear in context length, so FFN compute continues to dominate even at 28K tokens and the MoE discount keeps applying.

The full throughput sweep, NVLink, single-stream, Q8_0:

| Prompt tokens | Dense prefill (tok/s) | MoE prefill (tok/s) | Prefill speedup | Dense decode (tok/s) | MoE decode (tok/s) | Decode speedup |
|---:|---:|---:|---:|---:|---:|---:|
| ~520    |   862 | 2,252 | 2.61x | 23.9 | 112.0 | 4.69x |
| ~950    |   844 | 2,873 | 3.40x | 23.7 | 112.4 | 4.74x |
| ~1,835  |   920 | 3,278 | 3.56x | 23.0 | 108.3 | 4.71x |
| ~3,585  |   947 | 3,537 | 3.73x | 22.1 | 101.3 | 4.58x |
| ~7,085  | 1,089 | 4,058 | 3.73x | 21.3 |  98.3 | 4.62x |
| ~14,085 | 1,173 | 4,276 | 3.64x | 20.7 |  96.8 | 4.68x |
| ~28,085 | 1,158 | 4,198 | 3.63x | 20.4 |  94.1 | 4.61x |

**The memory story is the most underappreciated finding.** Both models were launched with identical parameters; llama.cpp's `-fit` mode then auto-sized context to fit available VRAM. The dense model was forced to shrink context from Gemma 4's native 262K window down to 104K — it couldn't fit even half the model's capability on 48 GB of consumer VRAM. The MoE runs at the full 262K with 11 GB of headroom. The mechanism is clean: half the layers means half the KV cache per token, and the global attention layers in the MoE use 2 KV heads instead of 4, which halves per-layer memory on exactly the layers where memory matters most. Per-cell KV cost on the MoE is roughly a quarter of the dense model's, which is how 2.5x more context cells fit in half the memory. Totals from `nvidia-smi`: 45.7 GB used for the dense model at 104K context, 36.3 GB used for the MoE at 262K context.

Memory breakdown at each model's maximum auto-fit context:

| | Dense (104K context) | MoE (262K context) |
|---|---:|---:|
| Model weights (GPU, both cards) | 31,109 MiB | 25,601 MiB |
| Embedding table (CPU-mapped) |  1,428 MiB |    748 MiB |
| Global attention KV cache     |  8,180 MiB |  5,120 MiB |
| Sliding-window KV cache       |  3,600 MiB |    900 MiB |
| Total VRAM used (both cards)  | 45,728 MiB | 36,282 MiB |
| Free VRAM headroom            |  3,424 MiB | 11,434 MiB |

**The topology finding from the previous post replicated cleanly — with one twist.** Running the same sweep with `CUDA_VISIBLE_DEVICES=0,1` forces the activation handoff across a PCIe 3.0 x1 slot instead of NVLink. On the dense model, this cost ~21% of prefill throughput at 28K tokens and was essentially invisible on decode. On the MoE, prefill took a similar ~24% hit at 28K — the smaller activation payload (hidden dim 2816 vs 5376) was offset by proportionally shorter compute, so the relative cost stayed in the same ballpark. But decode on the MoE showed a consistent ~6% PCIe penalty across every context length, where the dense model's decode was noise-level unaffected. The mechanism is that MoE's faster per-token decode (~10 ms vs ~48 ms) makes fixed PCIe handshake overhead — latency, kernel launch, sync — finally visible as a fraction of the per-token budget. The more general principle: when you accelerate compute, you expose communication costs that were previously hidden. Anyone planning small-active-param deployments on multi-GPU consumer rigs should factor this in.

**What this means for real deployment.** On paper the two models are in the same weight class. In practice they are in different deployment categories on this hardware. The dense model on two RTX 3090s is a single-user setup: 3 GB of headroom, 104K practical context, decode at speeds the user will feel. The MoE on the same hardware has 11 GB of headroom, the full 262K context window, and decode speeds in the range where it starts feeling like a cloud API. I haven't benchmarked concurrent load, so I won't claim measured multi-user numbers, but the combination of memory headroom and per-token speed suggests the MoE is a genuinely viable candidate for small-team shared inference where the dense variant is not. That's a future experiment, not a claim in this post.

**The one-line takeaway.** On two RTX 3090s with Q8_0 GGUF, the Gemma 4 MoE variant is not just "a faster small model" — it's a different deployment category from its dense sibling. 4.7x faster decode, 3.6x faster prefill at long context, and the full 262K context window instead of an auto-shrunk 104K. For anyone sizing a local or small-team deployment on consumer multi-GPU hardware, the MoE is the default choice, and the margin is wider than the nominal 25B-vs-31B comparison suggests.
