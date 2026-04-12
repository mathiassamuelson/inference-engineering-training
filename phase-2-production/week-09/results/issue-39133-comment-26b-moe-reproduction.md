## Independent reproduction on Gemma 4 26B MoE, same symptom on a sibling architecture

Adding confirmatory data on a related Gemma 4 variant. Same hardware class, same backend-forcing path, same qualitative symptom: KV cache capacity is much smaller than the architecture should allow if the hybrid-attention fields of the model config were being used for sizing.

### Environment

- vLLM: `0.18.2rc1.dev73+gdb7a17ecc` (commit `db7a17ecc`) — from the official [`vllm/vllm-openai:gemma4`](https://hub.docker.com/r/vllm/vllm-openai/tags) image, digest `sha256:0cb12dc964e1dace0a78aecd8905461d851b135db0690726f08550f7c4922834`, image built 2026-04-02
- Hardware: 2× NVIDIA RTX 3090 (SM 8.6, 24 GB each), tensor parallel across both
- Model: [`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`](https://huggingface.co/cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit)
  - Quantization (from `config.json` → `quantization_config`): `quant_method: compressed-tensors`, `format: pack-quantized`, `num_bits: 4`, `group_size: 32`, `observer: mse`, `strategy: group`, `symmetric: true`, `type: int`

### `vllm serve` arguments

```
cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit
  --tensor-parallel-size 2
  --max-model-len <varied across runs — see sweep>
  --gpu-memory-utilization 0.90
  --limit-mm-per-prompt '{"image":0,"audio":0}'
```

`kv_cache_dtype` is left at default `auto`, which resolves to BF16. All four runs produce the same backend-forcing log line reported in the original issue:

```
Gemma4 model has heterogeneous head dimensions (head_dim=256, global_head_dim=512).
Forcing TRITON_ATTN backend to prevent mixed-backend numerical divergence.
```

### Relevant model architecture (from `config.json` → `text_config`)

- `num_hidden_layers`: 30
- `num_attention_heads`: 16
- `num_key_value_heads`: 8 (SWA layers)
- `num_global_key_value_heads`: 2 (global layers)
- `head_dim`: 256
- `global_head_dim`: 512
- `sliding_window`: 1024
- `attention_k_eq_v`: `true` (per the [model card](https://huggingface.co/google/gemma-4-26B-A4B-it): "global layers feature unified Keys and Values")
- `layer_types`: 25 × `sliding_attention` + 5 × `full_attention`, interleaved 5:1 with the final layer being `full_attention`
- `max_position_embeddings`: 262144

This is a half-depth MoE sibling to the 31B dense: same 5:1 SWA-to-full ratio, same head dimensions, same sliding window. It exposes two additional global-layer KV compressions that the 31B dense parameters in the original issue don't mention: `num_global_key_value_heads: 2` (4× fewer KV heads than SWA) and `attention_k_eq_v: true` (K and V stored as a single tensor on globals).

### Observed: `max_model_len` sweep

Four runs, fixed `--gpu-memory-utilization 0.90`, varying only `--max-model-len`:

| `--max-model-len` | Available KV (per GPU) | GPU KV cache size |
|---:|---:|---:|
| 8,192 | 10.93 GiB | 95,472 tokens |
| 32,768 | 10.92 GiB | 95,440 tokens |
| 131,072 | 10.92 GiB | 95,456 tokens |
| 262,144 | 10.93 GiB | 95,472 tokens |

Across a 32× range in `max_model_len`, the reported KV token capacity varies by 32 tokens — exactly two paged-attention blocks of `block_size=16`. Per-token cost is constant; reducing `max_model_len` buys essentially no KV headroom. The KV pool size itself is constant across runs, as expected (pool size is governed by `--gpu-memory-utilization`, not `--max-model-len`).

### Per-GPU per-token cost analysis

Observed capacity of 95,472 tokens from a 10.93 GiB KV pool implies ~**122,925 bytes/token** per GPU.

For one attention layer under tensor parallelism with TP=N, the per-GPU per-token KV footprint in bytes is:

```
(num_kv_heads / TP) × head_dim × K_V_factor × sizeof(dtype)
```

where `K_V_factor = 1` for layers with `attention_k_eq_v` (unified K/V) and `2` otherwise.

For 26B MoE at TP=2, BF16, three candidate interpretations of how vLLM sizes the KV pool:

| Interpretation | Per-GPU bytes/token | Predicted capacity from 10.93 GiB pool |
|---|---:|---:|
| Full hybrid-attention exploitation (SWA capped at 1024, globals at max_model_len, `attention_k_eq_v` honored) | ~5,500 (amortized over 262K-token sequence) | ~2.13M tokens |
| Heterogeneous layers honored, no SWA cap (25 × SWA + 5 × global, `attention_k_eq_v` honored) | 25×4,096 + 5×1,024 = 107,520 | ~109,000 tokens |
| **All 30 layers treated as SWA-shaped, no SWA cap** | **30 × 4,096 = 122,880** | **~95,500 tokens** |

The observed 122,925 bytes/token matches the third interpretation to within **0.04%** — essentially within the precision of the 2-decimal-place "10.93 GiB" report. The observation is also nominally consistent with the second interpretation plus ~14% block-level overhead, but that overhead factor isn't otherwise evident and doesn't fit as precisely.

The tight match with 30 × 4,096 is consistent with vLLM's KV manager treating every layer as if it had the SWA-layer shape (`num_key_value_heads=8`, `head_dim=256`, separate K and V), sized at `max_model_len`. That's what a manager reading only the top-level `num_key_value_heads` and `head_dim` from `text_config` and applying them uniformly across all layers would produce.

### If the above is correct, four fields are not yet consulted

Under that interpretation, the Gemma 4 KV manager on this vLLM build is missing four pieces of per-layer variability exposed in `text_config`:

1. **`sliding_window` × `layer_types`** — SWA layers should cap at 1024 tokens rather than `max_model_len`
2. **`num_global_key_value_heads`** — global layers have 2 KV heads, not 8
3. **`global_head_dim`** — global layers use 512, not 256 (larger, but offset by item 2)
4. **`attention_k_eq_v`** — global layers store K and V unified (1 tensor per layer, not 2)

If the SWA-sizing question has a shared root cause with these, resolving it may require reading layer-type-specific shape information during KV pool sizing rather than just uniform top-level fields.

### Framing

This is confirmatory data on a sibling Gemma 4 architecture (MoE rather than dense). The core question in #39133 is unchanged: is vLLM's Gemma 4 implementation expected to exploit `sliding_window` / `layer_types` when sizing the KV cache? The 26B MoE reproduction suggests the scope may extend beyond the SWA-windowing question alone — the per-GPU per-token cost matches "all layers treated as SWA-shaped, uncapped" tightly enough that it's worth checking the three global-layer fields at the same time.

Not asserting the implementation is wrong — reporting observations on an additional Gemma 4 variant and asking whether the design intent matches the measured behavior.
