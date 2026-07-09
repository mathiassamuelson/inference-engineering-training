# Week 9 Summary — The sweep tool, TP vs layer-split, and the KV-sizing bug that paused the week

**Dates:** 2026-04-10 → 2026-04-12 (Days 1–3), plus the Day 4 re-test on 2026-05-17 after the upstream fix landed
**Model:** `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` (Gemma 4 26B A4B MoE) throughout; llama.cpp side `bartowski/google_gemma-4-26B-A4B-it-GGUF:Q4_K_M`
**Hardware:** NVLink pair GPUs 0+2 for both configurations
**Engines:** vLLM `vllm/vllm-openai:gemma4` (0.18.2rc1.dev73) Days 1–3; `vllm/vllm-openai:v0.21.0` Day 4; `ghcr.io/ggml-org/llama.cpp:server-cuda` Day 2

## TL;DR

Week 9 built the program's first shared benchmarking tool (`tools/throughput_sweep.py`), caught two would-have-corrupted-everything measurement bugs during its bring-up, reframed the planned "vLLM vs llama.cpp" comparison into the more honest "tensor parallelism vs layer splitting," and measured two clean single-request crossovers — then discovered that vLLM's KV cache manager was ignoring Gemma 4's hybrid attention entirely, making the memory side of the comparison a "framework + bug" measurement. The week **paused itself at Day 3** rather than publish conclusions on top of a known allocator bug: it reproduced the bug quantitatively, contributed the data to the existing upstream issue (vllm-project/vllm#39133), and stopped. Five weeks later, Day 4 re-tested against vLLM 0.21.0's Hybrid Memory Allocator: **9.3× KV pool capacity from the same VRAM**, single-request throughput unchanged (the bug was in pool sizing, not attention math), and one remaining 2× over-allocation identified from the sweep data alone. Concurrent load was never measured this week — that gap carried forward.

## The sweep tool and two methodology bugs (Day 1)

`tools/throughput_sweep.py` was written as a fresh generalization of Week 8's script (which stays frozen as a historical artifact): explicit `--backend` flag recorded for provenance, model name discovered from `/v1/models` with no silent defaults, per-completion timing via streaming (TTFT from first text-bearing chunk, decode from `usage` token counts), and self-describing JSON output carrying the script's own git SHA + dirty state, run UUID, and full config. First entry in the shared `tools/` directory — the seed of what later became the T repo's toolchain.

Two bugs were caught during smoke testing, both of which would have produced convincing-looking wrong data:

1. **Prefix-cache hits masquerading as prefill throughput.** Identical lorem-ipsum prompts (with the small size a strict prefix of the large) let vLLM's always-on prefix caching serve 71.2% of "prefill" from cache — apparent prefill rates up to ~29,000 tok/s that were really cache-lookup speed. The tell: a 4× prompt-size increase producing only a 1.5× rate change. The proof: vLLM's own `Avg prompt throughput` engine counter matched the *uncached* fraction of sent tokens exactly (738 = 738). Fix: unique nonce per request, prompt built per-iteration. Post-fix, engine-counted tokens matched sent tokens within one token.
2. **Tokenizer heuristic overshoot.** A hardcoded 0.75-tokens-per-word ratio built every prompt 1.33× oversize (the constancy of the ratio across buckets was the systematic-not-noise tell). Fix: two-request startup calibration measuring `chars_per_filler_token` (5.986 for this model) and nonce token cost (11), both recorded into results metadata.

Decode measurements were unaffected by bug 1 (the `wall_time − ttft` subtraction isolates decode from cache state) and were tight throughout (stdev < 0.5 tok/s per bucket).

**Day 1 sweep (vLLM TP=2, single request):** prefill peaks ~9,600 tok/s at 2–4K tokens and declines to 4,614 at 65K — not monotonic; under-utilization below the peak, global-layer O(n²) above it. Decode declines monotonically 136.7 → 85.7 tok/s (512 → 65K). The **131,072-token bucket stalled**: request accepted, queued, never admitted to execution (`Running: 0, Waiting: 1`, 600 s client timeout) — a scheduler-admission stall, cause not identified, recorded as an unresolved data point.

## The reframe, and TP vs layer-split single-request (Day 2)

Day 2 opened by declaring the planned "vLLM vs llama.cpp" comparison ill-posed: Week 8's llama.cpp data was a different bit width (Q8_0 vs AWQ-INT4) and a different model (31B Dense vs 26B MoE), and the real question on this hardware is **which parallelism strategy wins on the NVLink pair** — the frameworks are vehicles, not the subject. Day 1's journal was left as written; the reframe is documented as Day 2's first event. The matched comparison: vLLM TP=2 AWQ-INT4 vs llama.cpp layer-split Q4_K_M, both ~17 GB on disk, both on GPUs 0+2, KV precision verified matched (BF16 vs F16 — same storage and bandwidth cost).

A new cross-check earned its keep immediately: llama.cpp's server-side `timings` block (captured into schema v2 output) showed the script's TTFT-based prefill rate running **20–33% below** the server's internal measurement — most plausibly SSE buffering on the prefill→first-token transition. Decode agreed within 0.5%. Consequence: llama.cpp prefill uses `server_timings`; **vLLM has no equivalent signal, so its prefill rates are lower bounds by an unknown, probably smaller amount** — an explicitly carried confound.

**Two crossovers (single-request, temperature 0):**

| | short context | crossover | long context (65K) |
|---|---|---|---|
| **Decode** | TP +6–9% (≤4K) | **~8K — tied at 118–119 tok/s** | layer-split +19% |
| **Prefill** | TP 3.0–3.1× (≤2K) | **~32K — tied** | layer-split +14% |

TP's decode declines 37% across the range; layer-split's only 18%. For the statmon-ai 7–15K window the configurations were effectively interchangeable on single-request work.

Carried as stated in the daily, not resolved here: (a) AWQ/Marlin vs Q4_K_M kernel quality is a residual confound no experiment on this hardware can cleanly separate; (b) **naive TP theory says TP should win decode, and the data disagrees sharply** — three candidate mechanisms were recorded (hybrid attention shrinking the sharded-KV-read benefit; vLLM per-step overhead; all-reduce cost per decode step), all flagged as hypotheses, none tested.

Side observations that held up: the 5:1 SWA-to-global layer ratio is a Gemma 4 family property (25/5 on the MoE, 50/10 on the dense); hybrid attention made llama.cpp's 131K KV cache cost just 2.86 GB; and **vLLM's KV allocation looked ~5.5× more expensive per token of capacity than llama.cpp's** (≈120 KB vs ≈22 KB/token) — the observation that became Day 3.

## The KV-sizing bug: reproduce, contribute, pause (Day 3)

Step zero of the planned source-dive — a search of the upstream tracker — found **vllm-project/vllm#39133**, filed five days earlier against the 31B Dense on identical-class hardware, asking exactly Day 3's question. The day pivoted from investigation to reproduction-and-contribution.

The empirical core: four launches at `--max-model-len` ∈ {8K, 32K, 131K, 262K} showed the KV pool **flat at ~95,472 tokens across a 32× MML range** (varying by exactly two 16-token blocks). Per-token arithmetic against the checkpoint's `text_config` discriminated three candidate interpretations; observed 122,925 B/token matched "**all 30 layers sized as SWA-shaped at full MML, no window cap**" within 0.04% — an exact architectural calculation with no free parameters. The reading: four config fields (`sliding_window`×`layer_types`, `num_global_key_value_heads`, `global_head_dim`, `attention_k_eq_v`) were not consulted during pool sizing — a broader finding than the issue's original framing. Posted as a confirmatory comment with environment pinning and the sweep table (archived at `results/issue-39133-comment-26b-moe-reproduction.md`).

**The close-out call:** Day 2's crossovers were declared suspect — the oversized allocation plausibly penalizes TP and layer-split differently — and publishing or running concurrent benchmarks on the buggy build would produce numbers whose interpretation was provisional. Week 9 closed at Day 3 with the TP-vs-layer-split investigation **paused, not abandoned**, scripts and methodology ready for a re-test when the fix landed. Week 10 was re-pointed at production-deployment concerns (observability, front door) partly to stay on Gemma 4 and keep the re-test cheap.

## The re-test: vLLM 0.21.0 HMA before/after (Day 4, 2026-05-17)

vLLM 0.21.0 landed the Hybrid Memory Allocator. The same four-point MML sweep, same VRAM budget:

| `--max-model-len` | old build KV tokens | 0.21.0 KV tokens |
|---:|---:|---:|
| 8,192 | ~95,472 (flat) | 146,668 |
| 262,144 | ~95,472 (flat) | 891,535 — **9.3×** |

A two-parameter linear-in-MML model (SWA layers capped at their 1024-token window; global layers scaling with MML) fit all four points within 0.1%, and the fitted global-layer cost of exactly 2,048 B/token/GPU supports a structural reading, stated in the daily with confidence proportional to that fit:

- HMA **correctly caps SWA layers and sizes globals at true global dimensions** (2 heads × 512 head_dim).
- HMA **does not apply Gemma 4's `attention_k_eq_v` unification** — K and V are stored separately on global layers despite the architecture declaring them identical. A clean 2× over-allocation on globals remains; exploiting it would roughly double the pool again (~1.7M tokens at MML 262K). Identified from sweep data alone; whether it's a deliberate constraint or an unimplemented optimization was left open.
- A **~400 MiB fixed per-sequence overhead** is clearly present in the fit and dominates amortized cost at small MML; its source was not identified.

**Single-request throughput was essentially unchanged across the fix** (decode 160.2 → 94.6 tok/s over 512→65K, a 41% decline vs the old build's 37% — within run-to-run noise; prefill ceiling ~9,850 tok/s, same shape). Day 4's interpretation: #39133 was an allocator-*bookkeeping* bug — attention computation already respected each layer's true pattern, so per-step bandwidth was always correct, and what HMA unlocks is the concurrency ceiling, not single-stream speed. The before/after story therefore lives in concurrent benchmarking, which this week did not run.

**Prediction miss (preserved):** the pre-test prediction was that HMA would reduce per-token cost as MML *increased* via freed SWA over-allocation shrinking. The observed shape is the opposite — capacity grows monotonically with MML because SWA cost is fixed and only globals scale. The daily records the wrong prediction explicitly.

## Not measured this week

- **Concurrent load** — deferred at Day 1 (script has no concurrent dispatch), blocked at Day 3 (buggy build), explicitly deferred again at Day 4 per one-experiment-at-a-time. The 17.90×/3.40× (Day 4) and 24.24×/3.91× (old build) concurrency figures are engine capacity math, and Day 3 logged a rule about them: the internally computed "max concurrency" metric was removed from the issue comment because its computation hadn't been traced.
- **True vLLM prefill rates** — no server-side cross-check exists on vLLM; all vLLM prefill numbers this week are TTFT-based lower bounds. Prometheus-based measurement was proposed and not done.

## Open as the week left them

- **Whether the Day 2 crossovers survive the fix.** Day 4 established that single-request throughput didn't move, but the TP-vs-layer-split question was paused pending concurrent measurement and was not re-adjudicated in these journals.
- **The decode-slope divergence** (layer-split flatter than TP at depth) — three hypotheses on record, none tested.
- **The 131K single-request scheduler stall** (Day 1) — never diagnosed.
- **K=V unification on globals** and the **~400 MiB fixed overhead** — both identified, neither root-caused.

## Methodology lessons logged this week

- **Cross-check every benchmark against a server-side metric the script didn't compute.** The engine's own token counter caught bug 1; the `server_timings` block caught the 20–33% TTFT skew.
- **Rates that scale suspiciously with input size are measuring the wrong thing** — in either direction.
- **Tokenizer ratios are model-specific; calibrate, don't hardcode.**
- **Search the upstream issue tracker before starting a novel investigation** — a 2-minute search replaced hours of duplicative source reading.
- **Exact architectural arithmetic on one clean measurement can beat a noisy multi-point fit** — the 0.04% match had no free parameters.
- **Don't publish on top of a known bug; pause and re-test.** The week's defining call, and the re-test made the bug's before/after a finding in its own right.
- **"One experiment at a time" ≠ minimal change per session** — Day 1 bundled five items because all five were prerequisites of one honest measurement.

## Artifacts produced this week

- `tools/throughput_sweep.py` — schema v1 → v2 (`server_timings`, `cached_tokens` capture); nonce defeat, startup tokenizer calibration, self-describing provenance metadata
- Sweep JSONs: vLLM old build (`...20260410T020153Z`), llama.cpp Q4_K_M (`...20260411T234207Z`), vLLM 0.21.0 (`...20260517T175943Z`)
- `results/week09-vllm-021-kv-sweep/` — four startup logs + extracted KV lines
- `results/issue-39133-comment-26b-moe-reproduction.md` + the live comment on vllm-project/vllm#39133
- Per-day journals (Days 1–4)

## Carried forward

1. **Concurrent benchmarking on the fixed build** — the extension of `throughput_sweep.py` and the measurement that resolves both the before/after story and the paused TP-vs-layer-split question. (Picked up in Week 11 Day 1.)
2. **Week 10 pivot** — production-deployment concerns (observability first, then nginx front door), with the `proxy_buffering` streaming footgun pre-flagged.
3. Known script debt: `cached_tokens` warning threshold too strict (fires on the BOS token).
