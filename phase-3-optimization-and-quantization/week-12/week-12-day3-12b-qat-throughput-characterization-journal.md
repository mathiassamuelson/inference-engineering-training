# Week 12, Day 3 — 12B QAT throughput characterization and the production MML decision

**Model:** `google/gemma-4-12B-it-qat-w4a16-ct` (12B instruction-tuned, quantization-aware-trained, 4-bit weights / 16-bit activations)
**Config under test:** single RTX 3090 (GPU 1, uuid `a7370cb3`), `vllm/vllm-openai:gemma4-unified` @ `sha256:e828735f...63ed450`, max-model-len 131,072, gpu-memory-utilization 0.90, CUDA graphs on
**Launcher:** `tools/start-12b-qat.sh` (carries the `gemma4_unified.py` patch mount and the full `--hf-overrides` blob from Day 2)
**Results files:** seven JSONs under `results/`, all `throughput_sweep_vllm-openai_gemma-4-12B-it-qat-w4a16-ct_c*_20260612T*.json`

## Goal

Day 2 proved the 12B QAT *fits* on one 24 GB card across its entire defined context range. Day 3 measured how it *performs*: throughput under concurrent load across the realistic Operator Copilot context range (~8–64K tokens), plus a long-context probe near the top of the serving window. The deliverable is a production max-model-len (MML — the maximum combined prompt + output length the server will accept) for the sub-agent worker, chosen from data rather than habit.

## Boot and verification

One boot for the whole session at MML 131,072 — the recipe's pin, and 2× the top of the realistic context range, leaving headroom for outliers. Verification matched Day 2 exactly:

- Placement confirmed on GPU uuid `a7370cb3` via `nvidia-smi --query-compute-apps` (22,152 MiB resident)
- KV cache pool: **10.99 GiB / 467,959 tokens** — identical to Day 2's figure at this rung, reconfirming that the pool does not shrink as MML rises on this model
- Init 47.6 s (21.1 s compilation)

One operational discovery: the launcher maps the GPU-1 worker to host port **8001** (8000 + GPU index), not 8000. Sensible for the eventual two-worker layout, but every client — probes, sweep tool, future orchestrator config — must target the right port. Cost a few minutes of "connection refused" debugging.

## A probe methodology lesson: instruction-tuned models need the chat template

The first warmup probe went to the raw `/v1/completions` endpoint and returned 200 tokens of "1111…" — output that looks exactly like a broken model (Day 1's silently-disabled-quantization failure was fresh in mind). It wasn't. The model is *instruction-tuned*: fine-tuned to respond to requests only when they arrive wrapped in its chat template (special turn-marker tokens like `<start_of_turn>user`). The raw completions endpoint sends text verbatim with no template, a distribution the fine-tuning pushed the model away from handling. At temperature 0, with digits in the prompt to latch onto, greedy decoding locked into a repetition loop.

The differential test — the same question through `/v1/chat/completions`, which applies the template server-side — produced a perfectly coherent answer. Debugging signature worth remembering: **degenerate repetitive output from an instruction-tuned model is a missing-chat-template symptom before it is a broken-weights symptom.** Test the chat endpoint before tearing into the config.

Corollary checked and accepted: `tools/throughput_sweep.py` itself uses the raw completions endpoint with filler prompts. That is fine for throughput work — token generation *speed* is content-independent, degenerate output reliably runs to the token cap (uniform wave sizes), and Week 11's 31B baselines used the same path, keeping methodology comparable. Coherence checks are a separate concern and use the chat endpoint.

Warm decode check after the probes: 467 tokens / 5.88 s = **79.4 tok/s**, matching Day 2's ~78.

## Sweep 1 — light end: concurrency 1–8 at ~8K prompt tokens

Four sequential runs of `throughput_sweep.py` (schema v3; 1 warmup + 3 measured waves each; 512 generated tokens per request; unique nonces per request, no `cached_tokens` warnings fired). Replication across waves was airtight throughout the day (±0.1 tok/s).

| c | Aggregate gen tok/s | × c=1 | Per-request decode tok/s | Wave wall |
|---|---|---|---|---|
| 1 | 48.3* | 1.00 | 69.6 | ~10.6 s |
| 2 | 72.5 | 1.50 | 55.7 – 67.9 | 14.1 s |
| 4 | 96.0 | 1.99 | 31.4 – 62.6 | 21.3 s |
| 8 | 112.4 | 2.33 | 16.4 – 50.4 | 36.5 s |

\* c=1 aggregate computed as generated tokens / total wall for comparability; the tool reports decode rate directly at c=1. The "aggregate" figure is system-level: total generated tokens divided by wave wall-clock *including* prefill — distinct from per-request decode rate, which excludes it.

**Prediction vs outcome.** Predicted c=8 aggregate at 4–5× the c=1 figure, based on decode batching amortizing weight reads. Actual: 2.33×. The error was a workload-shape blind spot: at 8K prompt / 512 output, each request brings sixteen tokens of *prefill* work per token of generation. Prefill — chewing through the prompt to build the KV cache (the stored attention state for already-processed tokens) — is the contended resource, not decode. Aggregate throughput asymptotes toward the prefill rate, and the prediction modeled the wrong bottleneck. (A mid-sweep revised prediction of 110–120 tok/s, made after seeing c=2 and c=4, landed.)

The per-request decode spread tells the same story from the other side: in vLLM's continuous batching, a request admitted late spends its decode phase sharing the GPU with siblings' chunked prefills. The c=8 floor of 16.4 tok/s is a request decoding through ~26 s of accumulated sibling prefill traffic.

## Sweep 2 — heavy end: concurrency 1–4 at ~64K prompt tokens

c=8 was dropped by design: 8 × 64K = 512 K tokens would press against the 468 K pool and measure scheduler-queueing behavior, not throughput.

| c | Aggregate gen tok/s | × c=1 | Per-request decode tok/s | Wave wall |
|---|---|---|---|---|
| 1 | ~9.0* | 1.00 | 51.7 | ~57 s |
| 2 | 9.8 | 1.09 | 9.3 – 51.0 | 104 s |
| 4 | 10.3 | 1.15 | 3.4 – 46.0 | 200 s |

\* computed as above for comparability.

**This is the day's headline.** At 64K context, concurrency buys almost nothing — 15% aggregate gain at c=4 — while the late-admitted requests are savaged (decode floor 3.4 tok/s: a request spending its whole decode phase underneath ~150 s of sibling prefill). Wave wall scales almost exactly linearly with concurrency (57 → 104 → 200 s ≈ serial prefill). **At the heavy end of the realistic context range, a single 12B worker is functionally a serial device.**

Production implication: at deep contexts, queueing requests at the front door yields nearly the same aggregate throughput as batching them onto the worker, with far better per-request latency behavior. This feeds directly into the Week 13 front-door (nginx / orchestrator) design.

## The depth curve

Combining both sweeps' c=1 rows with the Phase 3 probe:

| Prompt depth | Prefill tok/s | TTFT | Decode tok/s |
|---|---|---|---|
| ~8 K | 2,480 | ~3.3 s | 69.6 |
| ~64 K | 1,382 | ~47 s | 51.7 |
| ~102 K | 1,083 | ~94 s | 46.2 |

(TTFT = time to first token, effectively the prefill duration at c=1.)

Smooth, no cliffs. **Prediction vs outcome:** predicted 64K prefill at 2,100–2,400 (actual 1,382) and decode at 58–65 (actual 51.7) — both misses in the same direction. Attribution: over-reliance on "sliding-window attention caps most layers." The global-attention layers attend over the *full* context during prefill, and that depth-quadratic term plus deeper KV reads on decode cost more than the global-to-SWA layer ratio naively suggests. The 102K predictions, extrapolated from the measured 8K→64K trend instead of from theory, landed (prefill predicted 1,100–1,300, actual 1,083; decode predicted 45–50, actual 46.2). Lesson restated: once two empirical points exist, extrapolate from them, not from architecture intuition. Follow-up: pull the exact SWA/global layer counts from the model config rather than working from recollection. *(Resolved Day 4: the checkpoint config's `layer_types` field gives 48 layers — 40 sliding-window, 8 global, a 5:1 ratio matching the 31B's 50/10 pattern. The global layers are an even smaller minority than recalled, which only sharpens the attribution: 8 of 48 layers carrying full-context attention is enough to dominate deep-context prefill cost.)*

## Long-context smoke test at ~104K

A retrieval-flavored coherence probe via the chat endpoint: a unique fact ("the maintenance code is MAKWGOIC-7741") planted at position 0, ~104 K tokens of synthetic log filler, then a question asking for the fact back — the hardest retrieval geometry, since only the global-attention layers can span that distance.

**Pass.** Exact code retrieved, coherent one-sentence answer, no degeneracy. Wall 98.5 s, consistent with the measured prefill rate (104,549 tokens ÷ 1,083 tok/s ≈ 96.5 s). This is a smoke test, not a quality evaluation — the systematic long-context quality work remains a separate effort.

Two sizing stumbles along the way, same lesson twice:

1. The first probe attempt drew an HTTP 400 — the filler was sized by rough mental arithmetic to "~100K" but actually tokenized to ~135 K, over the 131,072 MML. Silver lining: the 400 confirms the server **enforces MML at admission with a clean rejection** rather than truncating or crashing — exactly the behavior a front-door router can rely on.
2. The resized "~99K" prompt actually tokenized to 104.5 K — the chat template plus real tokenization diverged ~5% from the raw-endpoint calibration constant.

Lesson: character-based token estimates are ±5–10% at best; only the server's reported count is real. The sweep tool gets this right by calibrating against the live server; ad-hoc probes sized by hand do not.

## Decision: ship the 12B sub-agent worker at MML 131,072

1. **It costs nothing in memory.** The KV pool is flat across MML rungs on this model (10.99 GiB regardless — Day 2 finding, reconfirmed at boot). Raising MML does not shrink capacity; this is not the 31B-style tradeoff.
2. **Concurrency headroom at the realistic range is ample.** The 467,959-token pool admits seven 64K requests or fifty-seven 8K requests simultaneously. Admission capacity is not the binding constraint.
3. **The binding constraint is prefill serialization, not memory.** What MML controls in practice is the worst-case time a single request can park the worker: ~94 s TTFT at the cap.
4. **The config is verified end-to-end at depth:** load (Day 2), throughput (today), and a passing retrieval probe at 104 K.

The alternative — capping MML at 65,536 to bound worst-case park time at ~47 s — buys latency protection at the cost of outlier headroom, and that protection belongs in the orchestrator anyway (request-size routing or admission policy at the front door, Week 13's domain). Today's clean MML-overflow 400 is precisely the signal that front door can act on.

## Answers to the session's open questions

1. **Throughput under concurrency across 8–64K:** light end scales to 2.33× aggregate at c=8 and is still rising slowly (prefill-bound asymptote); heavy end is effectively serial (1.15× at c=4) with severe per-request degradation. No true saturation plateau reached at the light end, but the slope at c=8 makes the asymptote clear.
2. **~100K request profile:** prefill 1,083 tok/s, TTFT ~94 s, decode-at-depth 46.2 tok/s, coherent output with successful long-range retrieval.
3. **Production MML: 131,072**, rationale above.

## Carry-forward notes

- Worker on GPU N serves host port 8000+N — encode this in all client configs.
- Degenerate "1111…" output from an it-model → check chat template before suspecting weights.
- Char-based prompt sizing is ±5–10%; trust only server-reported token counts.
- vLLM rejects over-MML prompts with a clean 400 at admission — usable as a routing signal.
- At deep contexts the worker is serial; front-door queueing ≈ batching for aggregate, far better per-request. Input to Week 13 design.
- Parked levers remain parked: fp8 KV cache, `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`, video-zeroing override — none was needed.
