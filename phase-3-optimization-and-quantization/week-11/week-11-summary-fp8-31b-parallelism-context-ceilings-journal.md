# Week 11 Summary — FP8 31B Dense on 4×RTX 3090: how to parallelize it, how far its context stretches, and why that points to a delegation architecture

**Phase 3 (Optimization & Quantization), Week 11**
**Model:** `RedHatAI/gemma-4-31B-it-FP8-block` (31B Dense, FP8 weights)
**Stack:** vLLM 0.21.0, pinned by digest `sha256:a230095847e93bd4…`, text-only, Docker, 4×RTX 3090 (GPUs 0+2 on a NVLink bridge, GPUs 1+3 on PCIe 3.0 x1)
**Dates:** 2026-05-30 → 06-09

---

## What the week was about

The week had one model and one machine, and asked three questions in sequence: **how should this model be split across GPUs, how far does its context window actually stretch, and what does that mean for the real deployment** (statmon-ai, which over the week broadened from a monitoring tool into an orchestrator agent that fans out investigation work to ephemeral sub-agents). The answers turned out to chain together — the parallelism finding set up the context finding, and the context finding is what makes the case for the orchestrator/sub-agent architecture rather than a single big server.

The single most important conclusion, stated once up front: **no single serving configuration of this model serves the use case well.** Tensor parallelism is fast but context-limited; pipeline parallelism has the context but isn't fast enough to be interactive. That tension is the week's real result, and it's what motivates the Week 12 architecture.

---

## The chain of findings, day by day

### Day 1 — tooling and the first hard constraint

Built the gating dependency: extended `tools/throughput_sweep.py` to a concurrency-aware load tool (schema v3, async `httpx`, a "wave" model where `--concurrency` sets wave width and `--iterations` sets measured waves). Aggregate throughput is computed from per-request records, not measured separately, and the script carries a documented caveat that per-request TTFT under concurrency >1 is a client-side event-loop artifact while the aggregate is immune. Also created `tools/start-vllm.sh`, the parameterized launcher used all week.

The first hard fact landed here, almost by accident: a smoke-test boot at MML 131072 failed **at KV allocation, not on weight load**. That single failure framed the entire week — the 31B Dense under TP=2 is *weight-dominated*. The weights eat most of each card, and KV cache is whatever's left. Every later result is downstream of that.

### Day 2 — the KV cost model, and the text-only win

Characterized the KV cache empirically (the 31B's KV cost doesn't transfer from the 26B MoE work in Week 8). The fitted model:

> **TP=2 per-sequence KV/GPU ≈ 1.97 GiB + 39.2 KiB/token × sequence length**

confirmed to ~1% across a 16× span of sequence length. This two-coefficient model became the week's workhorse — it predicted ceilings before we went looking for them, and it held up every time we checked it.

A mid-session decision paid off larger than expected: disabling all multimodal modalities (`--limit-mm-per-prompt '{"image":0,"audio":0,"video":0}'`) for the text-only deployment didn't just free the encoder cache budget — it dropped the **vision tower weights entirely** (15.85 GiB/GPU vs ~16.93 multimodal), raising available KV from 3.25 to 4.04 GiB/GPU. Text-only became the baseline for the rest of the week. Working MML was locked at 33024 (max-concurrency 1.25×); 65536 was dropped as out of reach at util 0.95.

### Day 3 — pipeline parallelism at PP=2 is non-viable

Tried to split the model as a 2-stage pipeline instead of a tensor-parallel pair. It failed at KV init: the binding stage exposed only 1.95 GiB of KV against a 3.22 GiB requirement for one full window. **Mechanism:** the 256K-vocabulary embedding and LM head *shard* under tensor parallelism (each GPU carries half) but do **not** shard under pipeline parallelism — they land whole on the first and last stages, inflating those stages' weights and starving their KV. A pipeline's usable pool is bounded by its tightest stage, so one starved stage caps the whole thing. PP=2's serviceable context (~4,624 tokens) came out ~12× smaller than TP=2's. On two GPUs, TP=2 is the only practical configuration.

This is also where a durable principle got written down: **KV cache is the residual** left after weights, CUDA-graph capture, overhead, and safety margin are all subtracted — so KV shrinks whenever any other component grows.

### Days 4–5 — PP=4 is viable, but doesn't beat TP=2 where it counts

PP=4 (all four GPUs, two stages on the NVLink pair, two on the x1 cards) is the only other practical option, since it's the one that actually halves per-GPU weights again. It boots cleanly with a large KV pool (160,048 tokens at MML 33024, ~3.9× TP=2's). The trade-offs:

- **Decode:** TP=2 wins everywhere, ~1.7× faster (e.g. at 32K: 34.9 vs 21.1 tok/s). The penalty is structural to 4-stage pipelining over a host bridge and *cannot* be steered away — naive and NVLink-steered placement were identical to three significant figures at c=1.
- **Prefill:** the two curves cross between 8K and 16K. TP=2's prefill *declines* with prompt length (per-layer all-reduce cost grows); PP=4's *rises* (no per-layer collective, the pipeline-fill bubble amortizes). By 32K, PP=4 prefills ~1.4× faster with ~7s lower TTFT.

Day 4 also produced a clean methodology lesson: a ~9.2s first-probe-after-boot spike masqueraded as a placement effect through two wrong conclusions before replication revealed it as a one-time cold-start cost — the empirical justification for `--warmup 1` on every probe.

Day 5 tested the natural thesis that PP=4's bigger pool would convert to higher throughput under concurrent load (c=4). **It was refuted.** TP=2 won aggregate generation throughput *and* fan-out completion time at every prompt size — even at the long-context end where its pool can't hold four full windows, where it serializes the requests and still wins. PP=4's pool is real *capacity* (it keeps ~4× as many long sessions resident) but at this concurrency capacity doesn't become throughput, and for synchronized fan-out it inverts. Capacity and throughput are different ceilings; the bigger pool buys the former, not the latter.

### Day 6 — the context ceilings, and the reframe that ties the week together

Found each config's maximum context length, with every claim confirmed by a real near-full-window request (a clean startup line is necessary but not sufficient — both attempted shortcuts produced numbers that died before serving).

- **TP=2 baseline (util 0.95): 54,496 tokens**, KV-bound. Decode ~33 tok/s, seconds-scale TTFT. Cost-model error −1.5%.
- **TP=2 with the CUDA-graph KV tax recovered (util 0.97): 66,848 tokens**, +22.7%, crossing the 64K tier that was impossible at baseline. Still KV-bound, still ~32 tok/s, seconds-scale TTFT. Cost-model error +0.7%. *Both recovery recipes the vLLM boot log recommends — disabling the estimate, or util 0.9907 — failed on this 24 GiB hardware (warmup OOM and init-reject respectively); the working path, util 0.97, was found by laddering.*
- **PP=4 (util 0.95): 262,144 tokens (256K)** — the model's full architectural max. **Architecture-bound, not KV-bound:** there was KV to spare at the wall (max-concurrency 1.30× at 256K). The pool grew with MML the whole way up because of sliding-window amortization — once a sequence exceeds the SWA window, the fixed window cost spreads over more tokens, so a fixed memory budget holds more of them. But it serves the long-context end at ~15 tok/s with a **~5-minute time-to-first-token.**

**The reframe.** The first cut of the Day 6 read called PP=4 the winner because it "covers the context need where TP=2 can't." That scored the wrong thing. For an interactive root-cause-analysis loop, a 5-minute TTFT fails the task no matter how much context loads. *Fit* is not the bar; *usable for the task* is. So the honest comparison is not "pick the config with more context" — the two configs bound different things:

```
config            ceiling     bound by        decode     TTFT @ ceiling   interactive?
TP=2 util 0.95     54,496     KV exhaustion    ~33 tok/s   seconds          yes
TP=2 util 0.97     66,848     KV exhaustion    ~32 tok/s   seconds          yes
PP=4 util 0.95    262,144     architecture     ~15 tok/s   ~5 minutes       no
```

---

## What it adds up to: a delegation architecture, not a bigger server

Neither config serves statmon-ai alone. TP=2 is interactive but context-limited (~67K recovered); PP=4 has the context but isn't interactive. That "neither suffices" result is exactly what motivates the Week 12 architecture — stop asking one config to be both:

- **Orchestrator: 31B TP=2 on the NVLink pair (GPUs 0,2)**, run at its interactive ceiling (~67K, tax recovered). Fast decode, seconds-scale TTFT. Holds investigation state and reasons over *distilled* findings rather than ingesting everything itself.
- **Sub-agents: a smaller model (candidate: the new Gemma 4 12B-QAT) on single GPUs (the x1 cards, 1 and 3)**, fanning out the bulk-context work — reading long log spans in parallel — and returning summaries.

PP=4's 256K is not the deployment target. It is the *evidence* that brute-forcing context onto the 31B does not produce an interactive system — which is the argument for delegation. The week's long-context measurement earns its place by ruling out the simpler single-config design.

---

## Principles and methods that held all week

- **One variable per boot; predict before measuring.** Every ceiling was predicted from the cost model, then measured, with the deviation logged. The model was never off by more than 1.5%.
- **A clean startup line is necessary but not sufficient** — every max-context claim got a functional probe. This caught the difference between an *advertised* pool and a *usable* one twice on Day 6.
- **n=1 timing probes are unreliable until replicated** (the Day 4 cold-start artifact).
- **Verify GPU placement by `nvidia-smi` UUID-join, never by launcher intent.**
- **Commit tool changes before any results-writing probe**, so the recorded git SHA matches the code that ran (the dirty-tree trap).
- **KV is the residual** after every fixed cost; it shrinks when anything else grows.
- **Capacity and throughput are distinct ceilings** — a bigger pool doesn't imply more tokens/second.

## Tooling produced or extended this week

- `tools/throughput_sweep.py` — schema v3, concurrency-aware, `--placement {naive,steered,na}` provenance tag, model identity captured throughout output.
- `tools/start-vllm.sh` — parameterized launcher; gained `--device-order` (Day 4, deterministic stage steering) and `--profiler-cudagraphs {on,off}` (Day 6, opt-in KV-tax recovery, default off-the-baseline preserving).
- Per-day journals and committed result JSONs under `phase-3-optimization-and-quantization/week-11/results/`, each config-tagged.

## Open items carried into Week 12 (planning only — not started)

1. **12B sub-agent viability is now the load-bearing question.** `google/gemma-4-12B-it-qat-w4a16-ct` was downloaded but failed to load on vLLM 0.21.0 (`gemma4_unified` architecture unsupported). The single-GPU load test is the first Week 12 task — the whole delegation architecture depends on it.
2. **vLLM version investigation** — find a release supporting both `gemma4_unified` and the 31B Dense path; evaluate 0.22.1 / K=V-unification (which would also lift the PP global-layer over-allocation noted Day 6). Coordinated upgrade with a 31B re-baseline; Week 11's numbers are the regression baseline.
3. **Adopt the tax-recovered util (0.97) as a baseline?** It buys +22.7% usable context on TP=2 but changes held-constant, so it's a deliberate choice requiring re-baselining — and the two boot-log-recommended recipes are non-viable here, so the working path (util 0.97) must be documented as the one that holds.
4. **Orchestrator/sub-agent box layout** — 31B TP=2 on the NVLink pair; two independent 12B-QAT workers on the x1 cards — now backed by this week's evidence rather than assumed.
