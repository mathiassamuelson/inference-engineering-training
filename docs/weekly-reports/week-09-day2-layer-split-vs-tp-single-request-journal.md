# Week 9 Day 2 — Layer-split vs tensor parallelism, single-request

## Reframe

Day 1 was planned as the first half of a vLLM vs llama.cpp comparison, with Week 8
Day 3 supplying the llama.cpp side (Q8_0 GGUF of Gemma 4 31B Dense). After Day 1's
vLLM AWQ-INT4 sweep on the 26B MoE, that comparison stopped making sense:

1. Different bit widths (Q8_0 vs AWQ-INT4) confound framework efficiency with
   quantization choice. A result saying "vLLM is faster than llama.cpp" would
   actually be saying "4-bit is faster than 8-bit," which we already knew.
2. Different model sizes (31B Dense vs 26B MoE) confound framework efficiency with
   architecture.
3. "Framework A vs framework B" is the wrong question for this hardware anyway.
   What I actually want to know is: given NVLink between GPUs 0 and 2 and PCIe 3.0
   x1 between all other pairs, which **parallelism strategy** wins — tensor
   parallelism across the NVLink pair, or layer splitting? The frameworks are
   implementation vehicles for those strategies, not the subject.

So Day 2 became a matched-footprint comparison of TP-2 (vLLM AWQ-INT4) against
layer splitting (llama.cpp Q4_K_M), both ~17 GB on disk, both on GPUs 0 and 2
over NVLink, same 26B MoE model.

The reframe is honest about itself: Day 1's journal entry was written under the
old framing. Rather than retroactively edit it, the reframe gets documented here
as the first event of Day 2, which is closer to what actually happened.

## Configuration

Both configurations target GPUs 0 and 2 (the NVLink pair). The AWQ-INT4 and
Q4_K_M weight files are both ~17 GB on disk. This is the matched-footprint
criterion — 4-bit weight representations at the same VRAM cost, even though AWQ
and Q4_K are not bit-equivalent kernels. The deployment constraint that binds
in practice is on-disk/VRAM footprint, so matching that is what matters.

**vLLM (TP=2):**

```
docker run --rm -it \
  --gpus '"device=0,2"' --ipc=host --shm-size 16G --network host \
  -e HF_TOKEN=$HF_TOKEN \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:gemma4 \
  cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":0,"audio":0}' \
  --host 0.0.0.0 --port 8000
```

**llama.cpp (layer split):**

```
docker run --rm -it \
  --gpus '"device=0,2"' --ipc=host --network host \
  -e HF_TOKEN=$HF_TOKEN \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/llama.cpp:/root/.cache/llama.cpp \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -hf bartowski/google_gemma-4-26B-A4B-it-GGUF:Q4_K_M \
  -c 131072 -ngl 999 \
  --split-mode layer --tensor-split 1,1 \
  -fa on --parallel 1 \
  --host 0.0.0.0 --port 8000
```

## KV cache precision — matched

Verified from server init logs on both backends.

- **llama.cpp:** `K (f16): 1280 MiB, V (f16): 1280 MiB` for the 5 global-attention
  layers; `K (f16): 150 MiB, V (f16): 150 MiB` for the 25 SWA layers. Both
  caches F16.
- **vLLM:** config dump shows `kv_cache_dtype=auto` with model `dtype=torch.bfloat16`.
  vLLM's documented behavior is to resolve `auto` to the model activation dtype, so
  KV cache is BF16. No override appears in the logs between config dump and KV
  allocation.

F16 and BF16 are both 16-bit floats with identical storage (2 bytes/element) and
identical bandwidth cost on Ampere. They differ in exponent/mantissa split, which
matters for generation quality, not throughput. Confound ruled out for this
comparison.

## Methodology — carried forward from Day 1

1. **Nonce-prefixed prompts** to defeat vLLM's default prefix caching. Without
   this, repeated iterations hit cached KV and measure lookup speed, not prefill
   compute.
2. **Tokenizer calibration at startup** (two throwaway requests) to get accurate
   `chars_per_filler_token` and `nonce_tokens` rather than a fixed heuristic. For
   Gemma 4 26B MoE via llama.cpp: `chars_per_filler_token=5.97`,
   `nonce_tokens=11`. Produced a target of 512 tokens hitting 507–511 actual —
   well-tuned.
3. **Cross-check against a server-side metric the script didn't compute.**

## Methodology — new in Day 2

### `server_timings` cross-check now in the JSON output

llama.cpp-server emits a top-level `timings` block on the final SSE chunk
containing `prompt_ms`, `prompt_per_second`, `predicted_ms`, and
`predicted_per_second`. These are computed server-side and don't include any
network or SSE framing overhead. `tools/throughput_sweep.py` now captures this
block into each iteration record as `server_timings`, prints the server-reported
rates to stderr alongside its own measurements, and bumps output schema to v2.
vLLM doesn't emit this block; the field is omitted in that case.

This caught an important effect on llama.cpp: the script's TTFT-based prefill
rate was systematically 20–33% lower than the server's internal `prompt_ms`
measurement. The decode rates agreed to within 0.5%. The most plausible
explanation is SSE buffering on the prefill side — `t_first_token` fires when
the first streamed chunk is flushed to the client, not when the server finishes
the prefill pass internally. Decode streams token-by-token so buffering is
amortized, but the prefill→first-decode-token transition happens once and its
latency inflates TTFT.

**Consequence for analysis:** on llama.cpp, treat `server_timings.prompt_per_second`
as the trustworthy prefill rate; the script's TTFT-based rate is a lower bound.
On vLLM, which has no equivalent server-side signal, the script's TTFT-based
rate is what we have, and it's probably also a lower bound by some unknown but
likely smaller amount (vLLM's streaming internals are different from llama.cpp's,
and the cross-check isn't available to measure the gap). This is a real
remaining confound for the comparison; noted below.

### BOS-token false alarm on `cached_tokens`

The sweep fired a `[warn] cached_tokens=1` warning on every measured iteration.
Turned out to be the BOS token: every prompt starts with `<bos>`, which the
server keeps cached across requests independent of the nonce-prefix strategy.
1 token out of 511 is 0.2% of prefill — well below noise. The warning threshold
in `tools/throughput_sweep.py` is too strict; should be tightened in the Day 3
extension to `cached_tokens > max(5, 0.05 * prompt_tokens)` so it only fires on
meaningful cache hits.

## Results

All numbers from the sweeps are single-request throughput at temperature 0,
256 generated tokens, 3 measured iterations + 1 warmup per prompt size.

| Actual tokens | vLLM prefill (tok/s) | llama.cpp prefill (tok/s) | Ratio | vLLM decode (tok/s) | llama.cpp decode (tok/s) | Ratio |
|---:|---:|---:|---:|---:|---:|---:|
| ~510   | 8,308 | 2,642 | TP 3.14× | 136.7 | 125.4 | TP 1.09× |
| ~2,042 | 9,635 | 3,210 | TP 3.00× | 132.5 | 121.6 | TP 1.09× |
| ~4,082 | 9,394 | 5,361 | TP 1.75× | 127.7 | 120.8 | TP 1.06× |
| ~8,166 | 8,853 | 6,691 | TP 1.32× | 118.7 | 119.1 | **tied**  |
| ~16,330 | 7,874 | 6,853 | TP 1.15× | 109.9 | 116.3 | LS 1.06× |
| ~32,660 | 6,392 | 6,426 | **tied** | 97.0  | 111.1 | LS 1.15× |
| ~65,320 | 4,614 | 5,263 | LS 1.14× | 85.7  | 102.0 | LS 1.19× |

(TP = tensor parallelism via vLLM, LS = layer splitting via llama.cpp.
llama.cpp prefill rates are `server_timings.prompt_per_second`. All other
values are from the script's own measurements.)

## Two crossovers

**Decode crossover at ~8K tokens.** Below 8K the TP-2 configuration has a
consistent 6–9% decode edge. At 8K both configurations produce 118–119 tok/s
— genuinely tied. Above 8K, layer splitting pulls ahead and the gap widens:
6% at 16K, 15% at 32K, 19% at 65K. TP-2's decode declines 37% across the full
range (136.7 → 85.7); layer splitting declines only 18% (125.4 → 102.0). The
two curves have very different slopes.

**Prefill crossover at ~32K tokens.** TP-2's prefill advantage is dramatic at
short contexts (3.0–3.1× at 512 and 2K tokens, where the comparison is almost
unflattering), narrows steadily as context grows, and disappears between 16K
and 32K. At 65K, layer splitting is 14% ahead. TP-2's prefill peaks at 2–4K
(~9,500 tok/s) and declines after; layer splitting rises monotonically until
16K (~6,800 tok/s) and then holds roughly flat.

## Operational window for statmon-ai (7–15K tokens)

- **Decode:** tied (118–119 tok/s at 8K; 110–116 at 16K, with layer splitting
  holding a small edge at the higher end).
- **Prefill:** vLLM TP-2 wins by 15–32%. Given how infrequently uncached prefill
  actually runs (prefix caching covers most real-world conversation continuations),
  the prefill advantage is worth less in production than the headline number
  suggests.
- Net: within this window, the two configurations are effectively interchangeable
  on single-request workloads. Framework choice gets decided by other axes —
  concurrent throughput, operational characteristics, stability — which is what
  Day 3 will measure.

## Remaining confounds

1. **AWQ-INT4/Marlin vs Q4_K_M are different 4-bit kernels.** Some fraction of
   the measured gap is kernel quality rather than parallelism strategy. I don't
   think this explains the crossovers — the crossover behavior at 8K (decode)
   and 32K (prefill) is more naturally explained by scaling effects than by a
   kernel-quality difference that happens to flip sign at specific context lengths
   — but I can't separate the two cleanly without running a matched-kernel
   comparison, which isn't possible on this hardware (llama.cpp can't run AWQ,
   vLLM can't run GGUF efficiently).

2. **No server-side prefill cross-check on vLLM.** On llama.cpp the script's
   TTFT-based prefill was 20–33% low vs the server's internal measurement. If
   vLLM has a similar SSE-buffering overhead, the true vLLM prefill rates are
   higher than shown and the prefill crossover shifts later (closer to 65K or
   past it). A principled next step is to use vLLM's Prometheus metrics
   endpoint — which exposes per-request prefill/decode timing independent of
   the SSE framing — to measure this gap and correct the comparison. Planned
   for Day 3 or Week 10.

3. **Naive TP theory says TP should be the decode winner.** Tensor parallelism
   shards attention heads across devices, so at decode time each GPU reads only
   its slice of the KV cache for each generated token, and the NVLink interconnect
   handles the all-reduce cheaply. Layer splitting, by contrast, serializes decode
   across devices — the activations have to ship from GPU 0 to GPU 2 after the
   first half of the layers and back again in a pipeline fashion. The data
   disagrees sharply: layer splitting's decode curve is flatter and wins above
   8K.

   I don't have a clean mechanistic explanation yet. Candidate explanations:

   - Gemma 4's hybrid attention means most decode work is on SWA layers with
     tiny caches, so KV bandwidth is less of a decode bottleneck than the naive
     picture assumes, and the benefit of sharded KV reads under TP shrinks.
   - vLLM's decode path may have per-step overheads (scheduling, allocation,
     CUDA graph capture-related) that cost more per generated token than
     llama.cpp's tighter decode loop, and this overhead dominates as the
     compute per token grows with context.
   - The NVLink all-reduce may be more expensive per decode step than expected
     once it's amortized over only a few tokens per batch.

   These are hypotheses, not findings. Worth a deeper investigation once the
   concurrent data is in — the single-request picture probably isn't where
   this question should get resolved.

## Side observations

### Gemma 4 layer-count ratio generalizes across model sizes

llama.cpp's KV allocation logs show 5 global-attention layers and 25 SWA layers
for the 26B MoE. This is the same 1:5 ratio as the 31B Dense (10 global + 50
SWA, from Week 8). The ratio appears to be a Gemma 4 family property, not a
per-size-class design choice. Useful for planning KV capacity on future Gemma 4
variants: expect global-attention layers to be ~17% of total and SWA layers to
be ~83%, regardless of parameter count.

### Hybrid attention makes long-context KV cheap

At `-c 131072`, llama.cpp allocated 2.86 GB total for KV cache — 2.56 GB for
the 5 global layers and 300 MB for the 25 SWA layers combined. The SWA layers
cap at 1,536 cells each regardless of context length; only the global layers
scale with `n_ctx`. At 65K (the sweep's largest prompt), KV footprint was
about half this. This is a ~6× reduction vs what standard attention would
cost at the same context length.

### vLLM KV allocation looks oversized

vLLM's KV allocation is ~5.5× more expensive per token of capacity than llama.cpp's: 10.93 GB for 95.5K tokens (≈120 KB/token) vs 2.86 GB for 131K tokens (≈22 KB/token). If vLLM allocates full-context KV for all 30 layers while llama.cpp allocates it only for the 5 global-attention layers, the expected ratio at large context is 30/5 = 6×. The observed 5.5× matches closely enough to make "vLLM isn't exploiting Gemma 4's hybrid attention" the leading hypothesis. Can't confirm from logs alone — needs a deeper look at vLLM's KV cache manager behavior on hybrid-attention models. Week 10 investigation candidate; potentially a real vLLM limitation worth a GitHub issue.

## Open questions for Day 3

1. **Does concurrent load flip the answer?** Day 1's Week 8 Day 4 data showed
   vLLM's continuous batching delivered 7× system throughput over the transformers
   library at batch=16. Layer splitting's serialization across devices should
   hurt under concurrency in a way it doesn't hurt at batch=1 — but llama.cpp's
   `--parallel N` flag enables its own continuous batching, and its behavior on
   a layer-split topology is unknown. This is the measurement that will actually
   answer the framework question for the operational window.

2. **vLLM prefill cross-check via Prometheus.** Measure the SSE-buffering gap
   on vLLM so the confound in the prefill comparison gets quantified rather
   than hand-waved.

3. **Decode slope divergence.** Not a Day 3 question (concurrent data probably
   obscures it), but the "layer splitting decodes flatter than TP at long
   context" result is surprising enough to deserve an explanation. Stretch for
   Week 10.

## Housekeeping

- Day 2 sweep results: `results/throughput_sweep_llamacpp_google_gemma-4-26B-A4B-it-GGUF-Q4_K_M_20260411T234207Z.json`
- Day 1 sweep results (for reference): `results/throughput_sweep_vllm-openai_gemma-4-26B-A4B-it-AWQ-4bit_20260410T020153Z.json`
- `tools/throughput_sweep.py` updated to schema v2 (captures `server_timings`
  and `cached_tokens`). Backward compatible — new fields are additive.
- `cached_tokens` warning threshold known to be too strict (fires on BOS);
  fix deferred to Day 3 when the script is extended for concurrent benchmarking.
