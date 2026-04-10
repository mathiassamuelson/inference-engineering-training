# Week 9 Day 1 — vLLM Gemma 4 MoE Throughput Sweep

**Model:** `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` (Gemma 4 26B A4B, AWQ-INT4)
**Backend:** vLLM 0.19.1rc0 (`vllm/vllm-openai:gemma4`, digest `sha256:0cb12dc964e1…`)
**Hardware:** 2× RTX 3090, tensor-parallel=2 over NVLink (GPUs 0+2)
**Serving config:** `--max-model-len 262144 --gpu-memory-utilization 0.90`

## Goal

Close the comparison gap from Week 8. Day 3 produced single-request throughput data for llama.cpp on this model; vLLM has no equivalent measurements yet. Day 1 of Week 9 is about getting vLLM data onto the same axes as Week 8 Day 3 so the two backends can be compared apples-to-apples on single-request throughput.

The prerequisite is a generalized throughput sweep script that targets both backends with a unified CLI and output schema, since Week 8's script was written against llama.cpp's request/response shape.

## What was built

Created `tools/throughput_sweep.py` as a fresh generalization rather than editing the Week 8 script in place — Week 8's copy stays frozen as a historical artifact. Design decisions settled up front:

- **Explicit `--backend` flag** (`llamacpp` or `vllm-openai`). Recorded in results metadata for provenance; does not switch code paths. Both backends are hit via the same OpenAI-compatible `/v1/completions` path.
- **Single file, not a shared-core + adapters split.** Start simple; split only if backend divergence makes it ugly.
- **Clean break from Week 8's output schema.** Future analysis code will be written fresh; no obligation to make old consumers still work.
- **Model name discovered from `/v1/models`** if `--model-name` is not explicitly passed. Exits with a clear error if discovery fails — no silent defaults.
- **Per-completion timing via streaming** (option B, not deferred). Wall-clock around the request, TTFT measured from the first text-bearing chunk, decode rate computed as `(completion_tokens - 1) / (wall_time - ttft)` using the token counts from the final chunk's `usage` block. Requires `stream_options.include_usage=true`, which vLLM supports natively. Backend-agnostic by construction.
- **Self-describing JSON output.** Metadata captures script git SHA + dirty state, run UUID, timestamp, hostname, platform, python version, backend, endpoint, model name + source (explicit/discovered), and full sweep config. No hardcoded model- or architecture-specific fields.
- **Output filename includes model slug and backend**, so runs against different models don't overwrite each other.

## Methodology bugs caught in flight

Two separate issues surfaced during smoke testing and were fixed before the real sweep ran. Both are worth remembering; both would have produced convincing-looking-but-wrong data if they'd gone unnoticed.

### Bug 1: Prefix cache hits masquerading as prefill throughput

Initial smoke test with 128- and 512-token prompts produced `prefill_rate_tok_s` values of ~7,700 and ~29,200 respectively. The 3.75× rate jump for a 4× prompt size jump was the first tell — prefill rate should not depend on prompt length like that if the measurement is honest. TTFT was ~22 ms in both cases, with only ~1.5 ms difference across a 512-token span. Back-solving gave an "asymptotic prefill rate" of ~360K tok/s, which is not plausible for a 4B-active MoE on dual 3090s.

The server log told the real story: `Prefix cache hit rate: 71.2%`. vLLM's automatic prefix caching was on by default (it's always on unless explicitly disabled), and because the script was sending the literal same `"lorem lorem lorem..."` prompt for every iteration of a given prompt size — **and** the size=128 prompt was a strict token prefix of size=512 — most iterations were serving KV blocks from the cache rather than doing real prefill work. The measurements were of cache-lookup speed, not compute.

Cross-check via vLLM's own `Avg prompt throughput` engine metric: 73.8 tok/s × 10 s window = 738 counted tokens. The script sent 2,562 total prompt tokens across the 6 requests in that window. At 71.2% hit rate, the uncached portion is 2,562 × 0.288 = 738. Exact match. vLLM's "prompt throughput" counts only actual prefill work, and it was reporting that roughly 29% of sent tokens actually got prefilled. The other 71% hit the cache.

**Fix:** prepend a unique 12-hex-character nonce to every request's prompt, and call `build_prompt()` inside the per-iteration loop rather than once per prompt size. vLLM's prefix cache hashes token sequences from the start of the prompt, so a unique prefix guarantees a cold prefill every time. Added ~11 tokens of overhead per request (measured later during calibration) — negligible at the prompt sizes being measured.

Re-smoke test confirmed the fix by cross-checking the same engine metric: 262.1 tok/s × 10 s = 2,621 counted tokens on ~2,623 sent. Within one token of a perfect match. Zero cache hits on the new requests.

The decode rate numbers were unaffected by this bug — decode happens after prefill regardless of cache state, and the `wall_time - ttft` subtraction cleanly isolates it. Decode rate was already tight (stdev under 0.15 tok/s) and stayed tight after the fix.

### Bug 2: Tokenizer heuristic overshoot

First full-range sweep revealed that every prompt size was coming in ~33% larger than requested. Size 512 → 693 actual tokens. Size 65,536 → 87,393 actual tokens. Ratio was almost exactly constant at 1.33 across every bucket.

Root cause: `build_prompt()` used a fixed heuristic `words_needed = target_tokens / 0.75`, which assumes ~0.75 tokens per word. With `"lorem "` being 6 characters per word, that works out to assuming 8 chars per token. Gemma 4's tokenizer actually produces ~5.99 chars per token for this filler text, so the heuristic built prompts with `8.0 / 5.99 ≈ 1.34` times more characters than needed — exactly the observed overshoot.

**Fix:** startup calibration. New `calibrate_prompt_parameters()` function sends two small throwaway `/v1/completions` requests at script startup:

1. A pure-filler sample (`"lorem " * 500`) to measure `chars_per_filler_token` for this specific tokenizer.
2. A sample nonce string to measure `nonce_tokens` — how many tokens the nonce prefix consumes.

Both values are recorded in the results metadata under `model.tokenizer_calibration` so results files self-document which calibration was used. `build_prompt()` now takes these two measured constants as arguments instead of relying on a fixed ratio. No new dependencies — just two extra API calls per script run, and the calibration is backend-agnostic because it uses the same `/v1/completions` endpoint both backends already expose.

Measured values for this model: `chars_per_filler_token=5.9860`, `nonce_tokens=11`. The 11-token nonce overhead is notable — Gemma's tokenizer doesn't have dedicated multi-character tokens for random hex strings, so each hex character tokenizes to roughly its own token. Other models will differ; calibration handles this automatically.

## Results

Full sweep with nonce-prefixed prompts and calibrated sizing:

| Requested | Actual tokens | Prefill tok/s | Decode tok/s |
|---:|---:|---:|---:|
| 512 | 511 | 8,308 | 136.7 |
| 2,048 | 2,044 | 9,635 | 132.5 |
| 4,096 | 4,086 | 9,394 | 127.7 |
| 8,192 | 8,174 | 8,853 | 118.7 |
| 16,384 | 16,346 | 7,874 | 109.9 |
| 32,768 | 32,692 | 6,392 | 97.0 |
| 65,536 | 65,384 | 4,614 | 85.7 |
| 131,072 | — | **stall** | **stall** |

Values are means across 3 iterations per bucket after 1 warmup iteration. Iteration variance was extremely tight — decode stdev under 0.5 tok/s at every bucket, prefill stdev under 1% of the mean. Raw per-iteration data in `results/throughput_sweep_vllm-openai_gemma-4-26B-A4B-it-AWQ-4bit_20260410T020153Z.json`.

Note that the requested sizes land on the right actual token counts now — the calibration is accurate to within a handful of tokens across four orders of magnitude.

## Analysis

**Prefill rate is not monotonic in prompt size.** It peaks around 2-4K tokens at ~9,600 tok/s, then declines steadily. Below the peak, the GPU is underutilized — there aren't enough tokens in flight to saturate the SM pipelines, so per-token rate is lower. Above the peak, global-attention's O(N²) cost starts dominating and per-token rate drops. This is the classic attention-compute curve; the earlier "linear fit" model I was sketching with only two data points was wrong to assume an asymptote.

Gemma 4's hybrid attention architecture (50 sliding-window layers at 1024-token window + 10 global layers) means the quadratic cost is concentrated in just 1/6 of the layers. That's probably why the decline is as gentle as it is — the 2× drop from peak (4K) to 65K would likely be much steeper on a pure-global-attention architecture at the same parameter count.

**Decode rate declines monotonically with prompt size**, from 136.7 tok/s at 500 tokens to 85.7 tok/s at 65K — a 37% slowdown across the measured range. This is a different mechanism than the prefill decline: each decode step has to attend over the entire prompt + generated-so-far, and attention read cost grows linearly with sequence length. The decode slowdown is the number that directly affects user-facing latency in long-context chat. For capacity planning, it means "tokens per second" is a range, not a constant, and the range depends on how much conversation history has accumulated.

**For the statmon-ai 7-15K operational range**, interpolating from the 8,174- and 16,346-token buckets gives approximately **8,900 tok/s cold prefill** and **115-118 tok/s decode**. The prefill number is largely academic for real chat usage — prefix caching will reduce the per-turn prefill cost to just the new tokens the user appended, so the cold rate only matters for the very first turn of a fresh conversation. The decode number is the steady-state user experience metric, and 115-118 tok/s for a 7-15K context is the practical answer to "how fast will this feel."

## The 131K stall

The 131,072-token bucket failed in both the pre-calibration and post-calibration runs. In the post-calibration run, the actual prompt was genuinely ~131K tokens (not inflated by the heuristic overshoot), so the failure isn't just "it was actually 175K and that was too big."

The client-side symptom is a 600-second read timeout on the warmup request. The server-side symptom from the previous run's log was `Running: 0 reqs, Waiting: 1 reqs, GPU KV cache usage: 0.0%`: the request was accepted at the HTTP layer, queued in the scheduler, and then never admitted to execution. That's a scheduler-queue stall, not a runtime failure. Prefill isn't slow; prefill is never starting.

I don't know the specific vLLM internal reason without digging. Candidate explanations include a `max_num_batched_tokens` limit refusing the request, some chunked-prefill interaction with the MoE path, or a KV-cache-block-allocation check that fails without surfacing an error. The `gpu-memory-utilization 0.90` setting should leave enough room for a 131K-token KV allocation on this model — roughly 2 GB at int4 KV + fp16 activations — but I haven't verified that.

This is worth investigating as a separate side-quest, probably framed as "what is vLLM's practical single-request ceiling for this model on this hardware under default configuration." But it's not blocking Week 9's main comparison work, because the 512-65K range already covers the statmon-ai operational window with headroom, and it's a range Week 8 Day 3's llama.cpp sweep also covered.

For today, accept the 131K as an unresolved data point and move on.

## Cross-check: server-reported vs. script-reported

vLLM's engine loggers report `Avg prompt throughput` and `Avg generation throughput` over 10-second windows. These are independent of what the benchmark script measures and provide a useful sanity check. Spot checks during the sweep were consistent with per-request measurements — e.g., during the 16K bucket, the engine reported ~8,700 tok/s prompt throughput, in the same neighborhood as the script's reported 7,874 tok/s prefill. The two metrics measure slightly different things (engine averages include scheduler overhead; script measures per-request TTFT), so they shouldn't agree exactly, but they agreed to within the expected gap.

The `Prefix cache hit rate` metric on the server is cumulative across the engine's lifetime, not per-run. Don't use it as a per-run signal — use `Avg prompt throughput` vs. sent-token-count arithmetic instead, which is what caught Bug 1 above.

## Deliverables landed

- `tools/throughput_sweep.py` — first entry in the shared `tools/` directory. Records its own git SHA + dirty state into every results file.
- `results/throughput_sweep_vllm-openai_gemma-4-26B-A4B-it-AWQ-4bit_20260410T020153Z.json` — raw sweep data, 7 clean buckets plus one recorded failure.
- This journal entry.

## Open questions and next steps

- **131K scheduler stall.** Understand what's actually refusing the request. Low-priority side-quest; not blocking Week 9's main arc.
- **Tokenizer heuristic calibration across models.** The current calibration is backend-agnostic but uses a lorem-ipsum filler, which may tokenize differently than real prompts for some tokenizers. For vLLM, this doesn't matter because we only care about hitting a target token count; but worth remembering if the script gets reused for experiments where prompt content matters.
- **vLLM vs llama.cpp single-request comparison.** Day 2's task. Run the same sweep against the llama.cpp Q8_0 path using the same script, then put the two datasets side by side. The prefill curve shapes should be informative — llama.cpp's MoE path is mature and might produce a higher peak; vLLM's path has the missing `(E=128, N=352, int4_w4a16)` tuned config shape and is running on a fallback kernel.
- **Concurrent-load benchmarking** is Day 3 or later. Day 1's script doesn't measure concurrency yet; that extension is deferred until we have the single-request comparison nailed down.
- **MoE autotuner attempt** against the missing shape is the Week 9 stretch goal and still open.

## Lessons

1. **Always cross-check benchmark results against a server-side metric you didn't compute.** Bug 1 would have gone unnoticed if the server log hadn't been available to compare against. The engine's `Avg prompt throughput` ended up being the ground-truth signal that exposed the prefix cache confound. For every future benchmark, identify a server-side metric independent of the script's own timing and verify they tell the same story.
2. **A 4× input size change that produces only a 1.5× rate change is almost certainly measuring the wrong thing.** Any time the rate appears to scale suspiciously with input size — too much or too little — that's an invitation to check what's actually being measured. In this case the rate was scaling *more* than it should, which was the smoking gun.
3. **Tokenizer ratios are model-specific; don't hardcode them.** The heuristic overshoot could have been avoided entirely if calibration had been in the design from day one. Caught it because the actual vs. requested prompt size ratio was so cleanly constant (1.33 across every bucket) that it was obviously systematic, not noise. The fix turned out to be simpler than the original heuristic — one calibration function, two throwaway requests, fully backend-agnostic.
4. **"One experiment at a time" doesn't mean "minimal change per experiment."** Day 1 ended up bundling script generalization, backend support, per-completion timing, prefix cache avoidance, and tokenizer calibration into one session. That sounds like scope creep, but all five were prerequisites for the actual measurement that made the day worth doing. The experiment was "measure single-request throughput for vLLM on Gemma 4 26B MoE"; the other four items were plumbing the experiment needed in order to produce honest numbers. The rule is about not chasing unrelated questions mid-session, not about artificial minimalism.
