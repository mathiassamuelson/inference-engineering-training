# Program Capstone — Inference Engineering Training, Weeks 1–16

**Repo:** `inference-engineering-training` (`docs/program-capstone.md`)
**Written:** July 2026, Week 16 (the program's concluding week)
**Claim trace:** every substantive claim in this document maps to a source in the committed claim
inventory
(`phase-3-optimization-and-quantization/week-16/week-16-session3-capstone-claim-inventory.md`),
which points into the weekly and phase summary journals and, through them, the daily journals.

---

## What this is

This document is the spine of a 16-week self-directed training program in AI inference
engineering, run January–July 2026 on a single machine: four NVIDIA RTX 3090s (96 GB VRAM total)
in a consumer Gigabyte B650 motherboard, Ubuntu 24.04, CUDA 12.6. The program started on two
GPUs; the other two arrived in Week 2, and an NVLink bridge joined GPUs 0 and 2 in Week 7 — the
single most consequential hardware change of the program. The other two cards sit on PCIe 3.0 x1
links, a constraint that ended up shaping the final architecture rather than limiting it.

The program was originally planned at 24 weeks, grew to 28 as real findings inserted work the
plan hadn't anticipated, and concluded at 16 when its capstone deliverable — a validated two-tier
serving architecture — was achieved in substance ahead of the paper plan. The unbuilt phases were
dispositioned, not abandoned; the plan document records where each piece went.

Three public repositories hold the record:

| Repo | Holds |
|---|---|
| `inference-engineering-training` (this repo, "R") | results, journals, captures, the curriculum |
| `ai-training-tools` ("T") | the benchmarking/eval toolchain plus its bundled inputs — one SHA pins code and inputs together |
| `inference-reference-stack` ("IRS") | the deployment stack: Compose, nginx front door, observability |

Until Week 16 the first two were named `rtx3090-ai-training` and `rtx3090-ai-training-tools`.
They were renamed so the program's identity anchors to the work rather than to the hardware that
happened to run it; GitHub's rename redirects keep every historical link and journal reference
live, and the old names are never reused. Journals written before the rename keep the old names,
per the program's never-rewrite rule.

## The arc in brief

Week 1 measured a vanilla-PyTorch baseline and found the number the rest of the program is
implicitly measured against: the transformers library plateaus at **~5,000 tokens/second** of
total throughput on a 3B model, no matter how large the batch. Sixteen weeks later, the same four
consumer GPUs run a **quality-validated, concurrency-proven two-tier delegation architecture** —
a Gemma 4 31B orchestrator model on the NVLink pair and two Gemma 4 12B worker models on the x1
cards, all served from one version-pinned vLLM image. All three are 4-bit
quantization-aware-trained (QAT) models — trained with quantization in the loop rather than
compressed after the fact — with the quantization measured as quality-lossless against their
full-precision parents for the system's use case, and the tiers measured as not degrading each
other under full concurrent load.

The three largest single numbers in the program are worth stating together, because none of them
is a model-level optimization: switching serving frameworks multiplied system throughput
**7.12×** on unchanged hardware (Week 5); moving one two-GPU model split from a PCIe x1 link to
NVLink multiplied the same deployment's peak throughput **9.53×** (Week 7); and an allocator
correctness fix upstream multiplied KV-cache capacity — the memory budget that determines how
much conversation context a server can hold — **9.3×** from the same VRAM (Week 9's epilogue).
Infrastructure multipliers dominated everything done to the models themselves.

---

## Phase 1 — Foundation and baselines (Weeks 1–4)

Phase 1 built the measurement floor with vanilla PyTorch, then demolished — one per week — the
naive expectations it started with.

**Week 1** established the baselines on Llama 3.2 3B under the Hugging Face transformers library.
Switching the weights from 32-bit to 16-bit floating point (FP32 → FP16) halved the memory
exactly but sped up single-request generation only 1.56× (54 → 84 tok/s), not the expected 2–3× —
because generation is memory-bandwidth-bound: producing each token requires streaming essentially
the whole model through the GPU from its memory, and at the measured 504 of 936 GB/s the compute
units spend most of their time waiting for weights to arrive. Growing the batch from 1 to 1,200
requests plateaued *total* throughput at ~5,000 tok/s while *per-request* throughput collapsed 84
→ 4.2 tok/s. A linear memory model (peak = 6.47 GB + 13.03 MB × batch size, R² 0.9999) was the
phase's cleanest artifact.

**Week 2** tried a standard model-optimization recipe from the deployment toolbox — export the
model from PyTorch to the ONNX interchange format, then compile it with NVIDIA's TensorRT
optimizer — and got a negative result worth keeping: for LLMs, the recipe made inference
*slower*. The measured failure has a precisely pinned cause: on Llama 3.2 1B, the converted model
ran at 0.44× the plain PyTorch FP16 throughput (81 vs 183 tok/s) because ONNX Runtime left the
weights on CPU and copied ~1.2 GB across PCIe on every forward pass — its own "Memcpy nodes"
warning confirmed it. The other attempts failed before they could be measured: the direct PyTorch
export died on a trace error the journal attributes to the model's dynamic operations (rotary
embeddings, the growing KV cache), and the 3B model's export ran out of system RAM. Whether a
correctly device-placed TensorRT engine would have beaten PyTorch was not investigated — the
recorded finding is narrower: the off-the-shelf conversion path fails LLMs on placement and
export mechanics, and kernel-level optimization is worthless when the weights sit on the wrong
device. Purpose-built serving frameworks exist to manage exactly those things — placement, KV
cache, generation — at the framework level.

**Week 3** mapped the motherboard's real topology — one CPU-direct x16 slot, three x1 slots (~1
GB/s), and no peer-to-peer support anywhere, meaning no GPU can read another's memory directly;
every inter-GPU transfer detours through system RAM — and evaluated the three ways to spread
inference across multiple GPUs against it. **Tensor parallelism** (every GPU holds a slice of
every layer, and all GPUs must synchronize — an "all-reduce" — after each one) died on the
interconnect: a single 32 MB all-reduce took 379 ms on this fabric, ruling the strategy out
entirely *on that pre-NVLink topology*. **Data parallelism** (independent complete model
replicas, one per GPU, no communication at all) reached 93.6% of ideal linear scaling across all
four cards with Llama 3.2 3B (7,422 tok/s total). **Pipeline parallelism** (a model's layers
split into sequential stages, one stage per GPU) cost 8–18% of single-GPU throughput on Llama 3.1
8B just for splitting the model — overhead from synchronization and from "bubbles," the idle gaps
where a stage waits for its predecessor, not from data volume. And PCIe x1 turned out to have no
measurable effect on single-GPU inference once the model's weights are loaded into the card's own
memory — the finding that later made the x1 cards productive as worker slots.

**Week 4** brought in vLLM, on the same Llama 3.2 3B as Week 1, and closed the phase with a
calibration: the expected revolution arrived as **~1.3× the transformers throughput**, at every
concurrency level (the plateau is hardware bandwidth, which no framework repeals — the ceiling
moved from ~5,000 to ~6,100 tok/s with the same shape), and the real value showed up elsewhere.
In mixed traffic, continuous batching — vLLM admits and retires requests from the running batch
dynamically, instead of processing fixed batches in lockstep — let short requests finish in 0.27
s alongside 5-second long generations: a 95% latency cut versus static batching, where every
request waits for the slowest in its batch. PagedAttention, vLLM's block-based manager for the KV
cache (the per-request working memory that stores attention state for every token of context, and
the thing that competes with model weights for GPU memory), needed 109 KB per token where
transformers consumed 261 KB — 2.21× more concurrent requests from the same GPU at every sequence
length. And the server survived 1,200 simultaneous requests without a single failure. The
capacity lesson: memory math said 1,200 users per GPU, throughput said 100–150, and a realistic
latency target (95th-percentile response under two seconds, the kind of SLA an interactive chat
product carries) said **25** — plan backward from the latency requirement, not forward from the
hardware.

Week 4 also corrected Week 1 in the open: the original per-token KV-cache figure for Llama 3.2 3B
(344 KB) had assumed every attention head stores its own keys and values; Llama 3.2 3B actually
uses grouped-query attention — several query heads sharing each key/value head, 8 KV heads rather
than 24 — so the true figure is 112 KB, which matched vLLM's observed pool sizing almost exactly.
The Week 1 journal was not rewritten; Week 4 carries the correction. That pattern — expected,
measured, corrected in the open — repeats through the program, and Phase 1's misses are preserved
as such: vLLM was expected at 5–10× (got ~1.3×), TensorRT at 1.5–2× (got 0.44×), PCIe x1 was
expected to hurt (it didn't, for resident serving).

*(Filing note: Week 4 is a Phase-1 week whose journal lives under `phase-2-production/` — a known
artifact the Week 14 reorganization deliberately left in place.)*

## Phase 2 — Production inference at scale (Weeks 5–10)

Phase 2 ran in two movements: framework-and-interconnect, then a deliberate pivot onto a
just-released model family.

**Weeks 5–7: the interconnect correction.** Week 5 scaled vLLM data-parallel across all four GPUs
— four independent vLLM servers, one per card, each holding its own complete replica of Llama 3.2
3B (the same model as the baselines, for continuity) — at 95.4% of ideal linear scaling (18,053
tok/s system throughput), and settled the serving-framework question — does the software that
*serves* the model, transformers versus a purpose-built inference engine like vLLM, actually
matter once the hardware is fixed? — with a controlled head-to-head on the same model: **7.12×
the system throughput of transformers** on identical hardware and workload, with near-identical
per-request latency — the entire advantage is continuous batching serving many requests at once,
not faster individual requests. Week 6 scaled up to Qwen 2.5 14B under 2-GPU tensor parallelism
and hit a wall: 316.5 tok/s peak system throughput, against 3,116 tok/s for Mistral 7B on a
*single* GPU. That is a ~20× cost-per-token penalty once the doubled GPU count is charged: 9.8×
less throughput from 2× the hardware, so 19.7× fewer tokens per GPU-hour. (The week's journal
dressed this in cloud pricing — a dollar per hour for one GPU against two dollars for two, giving
~11.2M tokens per dollar for the 7B versus ~0.57M for the 14B — but the dollar figure is a scale
factor that cancels: as long as cost tracks GPU count, cost per token is just the inverse of
per-GPU throughput.) Week 7 installed the NVLink bridge — verified by topology query to be on
GPUs 0↔2, *not* 0↔1 as assumed, the first instance of what became a standing rule — and re-ran
the identical 14B configuration on the bridged pair: **3,018 tok/s peak, 9.53× its own PCIe
result**, with near-jitter-free latency (0.6% coefficient of variation across 30 trials at
concurrency 128). The week's own framing of the correction is the phase's methodological
centerpiece: *the Week 6 finding was not a conclusion about 14B models; it was a conclusion about
14B models over PCIe x1.* The substrate you measure through is part of the measurement.

**Weeks 8–9: the Gemma 4 pivot.** When Google released the Gemma 4 family on April 2, the
curriculum was paused to deploy it within 24 hours — a bet that deploying a brand-new model on
its release day, while the serving ecosystem's support for it was hours old, would teach more
than the scheduled material. It did:

- A llama.cpp crash on Gemma 4 31B Dense prompts above ~5,400 tokens was narrowed to its exact
  boundary (5,482 tokens worked; ~5,600 crashed), reported upstream, fixed by the maintainers
  overnight, and verified the next day.
- The same benchmark trap was sprung twice, once per engine. Both llama.cpp and vLLM reuse
  previously computed prompt state when a new request begins with text the server has already
  seen — llama.cpp through its slot cache, vLLM through prompt-prefix caching that is on by
  default. Our synthetic benchmark prompts all began with the same framing text — our own
  construction, in plain sight — but we hadn't connected that to the engines' caching behavior.
  So the sweeps were partly measuring cache lookups rather than real prompt processing (what
  serving engines call "prefill"), and the distortion ran in both directions: Week 8's llama.cpp
  sweep underreported the true prefill rate (the corrected Gemma 4 31B Dense baseline came out
  ~18% higher), while Week 9's vLLM smoke test reported prefill rates far too fast to be
  physically plausible — the engine's own counters showed 71% of the "processed" tokens coming
  from cache. Once the connection was made, the fix was simple: prepend a unique nonce to every
  prompt so no two ever share a prefix. The durable rule it left behind: cross-check every
  benchmark against a counter the server computes itself.
- The motherboard's P2P reality surfaced: every non-NVLink pair reports peer-to-peer unsupported,
  so CUDA silently stages transfers through host memory — which is why moving a pipeline-parallel
  deployment off the NVLink pair cost 21–30% of prefill throughput where naive bandwidth math had
  predicted ~1.5% — a miss the journals keep on the books — and why Week 7's NVLink result was
  really about peer access existing at all, not just about bandwidth.
- A head-to-head between the family's two variants — the 31B Dense and the 26B-A4B
  mixture-of-experts (MoE), which routes each token through only ~4B of its 26B parameters —
  found the MoE decoding 4.6–4.7× faster at every context length and fitting the full 262K
  context window where the dense model could not, with the mechanism being per-token KV cost
  (layer count × KV heads per layer), not total parameter count.
- A six-failure vLLM bring-up ended the FP8-on-Ampere road (FP8 is an 8-bit floating-point
  quantization format; Ampere is the RTX 3090's hardware generation) and yielded the cleanest
  insight of the arc: FP8 *weights* and an FP8 *KV cache* have different hardware requirements,
  though they are usually discussed as one feature — weights work because a kernel can store them
  in FP8 and compute in 16-bit on any GPU, while writing the KV cache in FP8 needs a hardware
  cast instruction that only arrived in later GPU generations. AWQ-INT4, a 4-bit weight format
  with mature Ampere support, worked.

Week 9 built the program's first shared benchmarking tool, `throughput_sweep.py` — provenance
metadata, self-describing output, the seed of the eventual toolchain repo. With it, the planned
"vLLM vs llama.cpp" comparison was reframed into the better-posed question of tensor parallelism
versus layer splitting: the two ways of putting one model on the two bridged GPUs. Measured one
request at a time on the Gemma 4 26B-A4B MoE, the two strategies trade places as context grows —
tensor parallelism is clearly faster at short context, layer splitting overtakes it as prompts
lengthen, and the crossover comes at around 8K tokens for decode and around 32K for prefill. Then
the week discovered that vLLM's KV allocator was ignoring Gemma 4's hybrid attention entirely.
(Hybrid attention: most of the model's layers attend only to a short sliding window of recent
tokens, so their KV never grows past ~1K tokens; only a handful of layers see the full context.
An allocator that exploits this needs a fraction of the memory of one that doesn't.) vLLM was
sizing all 30 layers at full context — the observed bytes-per-token matched that hypothesis to
0.04%. The finding was contributed as a quantified reproduction to the existing upstream issue
([vllm-project/vllm#39133](https://github.com/vllm-project/vllm/issues/39133)) — and then the
program **paused itself mid-week** rather than publish conclusions or run concurrency benchmarks
on a known-buggy allocator. Five weeks later the fix landed upstream (vLLM 0.21.0's Hybrid Memory
Allocator) and the re-test showed the fixed engine holding **9.3× the KV pool capacity of the
buggy build** in the same VRAM, with single-request throughput unchanged: the bug was allocator
bookkeeping, not attention math, so what it had been suppressing was concurrency, not speed. The
crossover measurements were parked pending concurrent re-measurement and never re-adjudicated —
the record leaves them open rather than quietly rehabilitating them.

**Week 10** closed the phase by building where the program would live: the public
`inference-reference-stack` repo — vLLM, Prometheus and Grafana for metrics and dashboards, and
NVIDIA's DCGM exporter for per-GPU telemetry, all under Docker Compose, with engine images pinned
by digest (the exact image hash, not a movable version tag) as the reproducibility anchor. The
planned Triton serving layer was dropped on a concrete version conflict, with the decision
documented in the repo: the stack's architectural value lives above the engine. (The Triton
deep-dive, scheduled twice, never happened — an honest arc of a curriculum item repeatedly
out-competed by higher-value work; its one deployment, a Week 6 embedding model with 3.5× dynamic
batching, stands.)

## Phase 3 — The delegation arc (Weeks 11–16)

Phase 3 is where the program's threads converge into a system. The driving application, referred
to below as "the use case," is an **operator copilot for root-cause analysis of a distributed
system**: an interactive investigation loop in which an operator converses with the system, the
system reads large volumes of logs and telemetry, and answers must come back at conversational
speed — so it needs a large context window *and* fast responses *and* the ability to work several
leads at once. The full architecture argument lives in
**[`docs/delegation-architecture.md`](./delegation-architecture.md)** — the finalized Week 15
write-up. This section tells the program story and defers to that document for the architecture
itself.

**Week 11: the finding that reframed the program.** The Gemma 4 31B Dense (then in FP8
quantization, running TP=2 — tensor-parallel across the two NVLink-bridged GPUs; the notation
TP=N / PP=N for tensor- and pipeline-parallel degree recurs below) was characterized under every
viable parallelism layout. A simple cost formula, fitted from measurements, predicted every
subsequently measured memory ceiling within 1.5%: each sequence costs a fixed ~1.97 GiB of KV
memory per GPU — the sliding-window layers' KV, which is capped at the window size no matter how
long the context, plus base allocation — and then ~39.2 KiB more per token of context, which is
the model's 10 global-attention layers whose KV grows with the full sequence. PP=2 proved
non-viable, for a structural reason: the model's input and output vocabulary layers (huge, at a
256K-word vocabulary) get split across GPUs under tensor parallelism, but under pipeline
parallelism they must sit whole on the first and last stages — inflating those stages' weight
footprint and starving the memory the KV cache needs. The decisive comparison: TP=2 is
interactive — seconds-scale time-to-first-token, and a decode rate (the speed of token-by-token
generation) of ~33 tok/s — but runs out of KV memory at a ~54K–67K context ceiling; PP=4 reaches
the model's full 262K context but serves the long end at ~15 tok/s decode with a ~5-minute
time-to-first-token. The week's reframe: **fit is not the bar — usable for the task is.** Neither
configuration served the interactive use case, and that "neither suffices" result is what
motivated a delegation architecture — a capable orchestrator reasoning over distilled findings,
with cheap workers doing the bulk-context reading — instead of a bigger server.

**Week 12: the worker tier validated.** The Gemma 4 12B QAT checkpoint (the published set of
trained weights, `google/gemma-4-12B-it-qat-w4a16-ct`) loads and serves on a single 24 GB card
(8.28 GiB of weights) — after a debugging arc whose first out-of-memory failure turned out to be
self-inflicted (a config override had silently disabled quantization entirely, so the card was
asked to hold ~24 GB of full-precision weights instead of the 4-bit checkpoint's 8.28 GiB) plus
one genuine bug in the serving image, patched with a three-line backport of the fix that already
existed upstream — a temporary measure, removed as planned at the next upgrade of the serving
framework, vLLM. The card has no memory ceiling for this model: the KV pool holds the model's
full context window (262,144 tokens — the maximum its design supports) with room for slightly
more than two such sequences at once (2.16× max concurrency). Production context length was
pinned at 131,072 — the model's validation boundary at the time of the Week 12 validation, not a
memory limit. The throughput characterization produced the week's headline systems result: at 8K
context, serving eight concurrent requests raises aggregate throughput 2.33× over serving them
one at a time — but at 64K+ context the worker is **functionally serial**, gaining almost nothing
from batching. That shape fed directly into the design of the front door (the single nginx
endpoint that fronts the worker pair): at depth, queueing requests matches batching them, with
better latency.

**Week 13: both tiers onto one engine version, and the quality verdict.** Both tiers moved onto a
single pinned image (`vllm/vllm-openai:v0.23.0`), retiring every per-model workaround. On that
shared stack, the quantization question was answered properly: **at both tiers — the Gemma 4 31B
orchestrator and the Gemma 4 12B workers — the QAT W4A16 model (4-bit weights, 16-bit
activations) is quality-equivalent to its BF16 parent for this use case.** The measurement used
LLM-as-judge evaluation — a stronger model scoring the outputs, run pairwise (which of two
answers is better, with every comparison presented in both orders so the judge's position bias
cancels) and pointwise (absolute rubric scores) — with deterministic format checking kept
separate from the judge, on matched-provenance captures: both models answering the same
standardized test prompts ("probes") under identical configurations, with the exact tool versions
recorded. At the orchestrator, QAT and its BF16 parent (the unquantized 16-bit original) tied on
guardrail adherence — whether the model respects the operating constraints its system prompt
imposes — across all eight probes; at the workers, both scored 4.83–5.0 on the pointwise 1–5
rubric with strict output-format conformance on all six probes, across both worker components.
The claim is deliberately scoped: the quantization that makes the two-tier layout fit this
hardware costs no measured quality **for this use case** — the probes and rubrics test the jobs
the copilot actually performs, not general-purpose benchmarks. That scoping is the task-fit
principle (see the method section) applied to evaluation itself: the question worth answering was
never "is the 4-bit model as good in general?" but "is it as good at the job we hired it for?"
The same week measured the QAT checkpoint decoding 36–50% faster than the FP8 variant it replaced
and built the evaluation toolchain (judge, probe, contract checker, bring-up gates) that produced
the verdict.

The migration also quietly retired the constraint that had forced the delegation decision in the
first place. The QAT checkpoint roughly tripled the orchestrator's KV budget, so a single Gemma 4
31B now serves the validated 131K context envelope interactively — Week 11's ceiling, measured on
the FP8 variant, no longer binds on the production stack. The architecture stands anyway, on the
motivations that never depended on the ceiling: context management and token cost (workers absorb
the bulk tokens so the orchestrator reasons over distilled findings), and concurrency at depth (a
long-context orchestrator is nearly serial; parallel workers are not). The architecture document
carries this re-examination in full; Week 11's measurement stands in it as the historical record
of what forced the decision when it was made.

**Week 14: order, a clean win, and an honest null.** The toolchain and its eval inputs moved to
their own public repo, dissolving a provenance friction structurally: every tool now records the
*tool repo's* commit from any working directory, so results landing in the data repo never dirty
the SHA that pins what produced them. The week's one experiment answered the worker-tier
deployment question. The Gemma 4 12B QAT worker fits comfortably on one card — so is anything
gained by splitting it TP=2 across the NVLink pair anyway? **The answer was yes, at every load
level.** Against the single-card configuration, TP=2 delivered +47% on decode serving one request
at a time, +81% on prefill, and an aggregate-throughput advantage that *grew* with load, reaching
+72% at eight concurrent requests. PP=2 on the same two cards, by contrast, performed about the
same as one card. The win is explicitly an NVLink result, not a free-everywhere one — it rests on
the two cards being able to synchronize cheaply after every layer.

What shipped, though, is not TP=2 workers. Two *independent* single-card workers deliver more
total throughput than one TP=2 worker — by straightforward arithmetic on the measured numbers,
two cards each sustaining 112.4 tok/s at eight concurrent requests beat one TP=2 pair's 193.3 —
and the box has exactly one NVLink bridge, already committed to the orchestrator's TP=2, where
giving it up was unacceptable and buying a second bridge for hardware scheduled to retire would
have been wasteful. So the production layout stands as the delegation architecture defines it:
the Gemma 4 31B QAT orchestrator split TP=2 across the NVLink pair, and two Gemma 4 12B QAT
workers, one per x1 card, each serving independently. That is the exact configuration Week 15
then put under concurrent load. And the week's long-carried nginx hypothesis died honestly. The
front door balances the two workers with nginx's least-connections strategy (`least_conn`), and a
Week 13 probe had seen all eight concurrent requests land on one worker; the diagnosis was a
missing shared-memory `zone` — without one, each nginx worker process balances on its own private
connection counters. The fix was applied and load-tested, and produced **no measurable change**:
the pool was already splitting evenly without it, both halves of the prediction were wrong, and
the corrected mechanism (with connection counts tied at zero between short requests, `least_conn`
falls back to round-robin, which alternates evenly even on private state) is in the journal. The
fix was kept as documented-correct and zero-cost; its effect is unobservable on this symmetric
two-worker pool.

**Week 15: the operational proof.** The one claim the architecture still owed itself: that the
tiers stay out of each other's way *under load*, not just that three services boot. The
experiment was designed and its predictions committed — including a falsifiable mechanism commit
— before any measurement. The verdict: **neither tier slows the other down.** With the other
tiers held at sustained full GPU utilization by flood traffic, neither the orchestrator nor a
worker loses more than ~0.5% of decode throughput at any measured prompt size — a ~7× margin
inside the pre-committed 3% bar. The falsifiable commit held: degradation shrinks as context
depth grows, exactly what the proposed mechanism requires (the coupling is per-token CPU work on
the shared host, so long-context decoding — slower, more GPU-bound per token — does fewer host
handoffs per second and is *less* exposed, not more). The nginx front door spread the flood's
requests 680/663 across the two workers (50.6/49.4%) — an evenness the record credits to
least-connections balancing over two identical workers, not to Week 14's zone fix. And the
mechanism work ended with a correction and a narrowing: the hypothesized interrupt-rate channel
was first dismissed on the wrong evidence (the kernel's *softirq* CPU counter, which doesn't
capture GPU interrupt servicing), then measured on the right one (*hardirq*, which does: 0.000%
under load), and finally dispositioned as *"unsupported and unnecessary"* rather than "proven
absent," because the counters measure servicing time, not rate, and host-CPU contention already
accounts for the whole ≤0.5% coupling.

One error from that week is worth telling at full honesty. By Week 15 the record held half a
dozen ceiling-adjacent numbers for the orchestrator family — 33,024; 54,496; 66,848; 131,072;
193,837; 262,144 — and the experiment design confused them, anchoring its "near-ceiling" probe
point to ~54K, which matches the *retired FP8* variant's ceiling rather than the QAT stack
actually under test (193,837 — the recorded "off by ~3.5×"). The empirical boot-log read caught
it, and the probe point was relabeled to what it actually is: a large-context point at ~25% of
the true ceiling. Deep-context isolation (150K+) is therefore an extrapolation — a safe one,
since the measured degradation decays monotonically with depth — and is stated as one everywhere
it appears.

Week 15 closed with the delegation-architecture write-up finalized under the same contract as
this document (tagged claim inventory before prose; one remembered claim deliberately withheld as
unsupported by the record). The architecture doc's own closing states the validation chain:
**viable** on the target cards (Week 12), **quality-lossless** under quantization for its use
case (Week 13), **co-resident** — all three services serving concurrently on one host (Week 13),
**evenly load-balanced** at the front door (Week 15), and **interference-isolated** under load
(Week 15) — with the thesis substrate-neutral and the consumer-GPU system as its proof case, not
its content.

**Week 16** is the program's conclusion: the repo renames, the consolidation that produced the
weekly and phase summaries this document traces into, this capstone, a method write-up, and plan
closure.

---

## The method

The program treats its working method as a first-class outcome, on par with the running system.
Its elements, each of which can be checked against the record rather than taken on faith:

- **Predict before measuring, in writing, then score it.** The Week 11 cost model predicted every
  later ceiling within 1.5%. Week 15 committed a full prediction table — including a falsifiable
  commit whose failure would have invalidated the proposed mechanism — before any measurement,
  then scored all five predictions and logged four refinements as refinements rather than hits.
- **One experiment per session; one variable per boot.** Warm-up runs are standard because a Week
  11 cold-start artifact produced two wrong conclusions before replication exposed it.
- **Honest nulls, kept by name.** The nginx zone fix that fixed nothing (Week 14) and the
  softirq-then-hardirq interrupt arc (Week 15) are preserved as corrected mental models, not
  buried. Prediction misses from Week 1's expectations through Week 15's mechanism are part of
  the record.
- **Journals are never rewritten.** Corrections are appended — the Week 15 resolution and the
  Week 16 consolidation-journal corrections are worked examples. Historical documents keep the
  names and beliefs of their time.
- **Provenance is mechanical, not aspirational.** Engine images pinned by digest — the immutable
  content hash, not a movable version tag — from Week 10 on; every result JSON carrying the
  toolchain repo's commit SHA (anchored to the tool's own file path, so it is correct from any
  working directory); commit-before-run so a dirty tree never contaminates a recorded SHA; and
  self-describing result filenames.
- **Ground truth over declared intent.** GPU placement is verified on every boot by joining each
  GPU's hardware UUID to the process actually running on it (the NVLink bridge itself was
  discovered on different GPUs than assumed); ports are confirmed live (a stale port preset would
  have *faked* an isolation result in Week 15); memory ceilings are found by stepping the
  configuration up until it refuses, rather than trusted from configs — or from memory, as the
  Week 15 near-ceiling confusion demonstrated.
- **Task fit over headline capability.** A through-line visible across the record rather than a
  dated decision: Week 6's decision framework ("the smallest model that meets quality
  requirements"), Week 8's deployment-category framing and its choice of prompt engineering
  over fine-tuning, Week 11's usable-for-the-task reframe, the plan's Week 12 operating principle
  ("the highest-fidelity model that gives an acceptable context window"), the one measured
  instance — the 12B worker judged good enough *in absolute terms* for its focused extraction job
  — and Week 13's quality verdict itself, deliberately scoped to the use case's own probes and
  rubrics rather than general benchmarks. The bar is sufficiency for the use case, not
  biggest-available. (Stated as a through-line, not dated to a moment; and not as "smaller
  matched bigger" — no comparison against a larger model exists in the record.)
- **Engage upstream.** The day-1 llama.cpp segfault report (fixed overnight), the quantified
  two-architecture reproduction contributed to vllm#39133, and the Week 9 pause — refusing to
  publish on top of a known bug — are the program's public-facing discipline.

The paper trail itself — daily journals, weekly and phase summaries, committed claim inventories,
appended corrections, pinned provenance in every result — is a deliverable. Four LinkedIn Pulse
articles were published from the work (the 31B day-1 deployment report, the dense-vs-MoE
head-to-head, the LLM-as-judge quantization-quality method, and the
tensor-parallelism-on-one-card finding); `docs/linkedin/README.md` is the publication index.

## Scope, honesty, and what remains open

- **These numbers are a terminal characterization of a retiring topology.** The 4× RTX 3090 box,
  with its physically separate GPUs, makes host-side contention the only possible cross-tier
  coupling channel; a single-die successor platform measures a different mechanism, and its
  numbers must not be trended against these. What transfers is the thesis, the method, and the
  pattern of operational proof — not the values.
- **Open, carried as open:** the worker's 131K–262K context range fits in memory but is
  quality-unvalidated; vLLM's remaining K=V-unification gap (~2× KV over-allocation on global
  layers) and a ~400 MiB fixed per-sequence overhead are identified but not root-caused; the Week
  9 single-request crossovers were never re-adjudicated after the allocator fix; the
  asymmetric-backend probe that would make the nginx zone's effect observable was scoped but not
  run.
- **Not measured, stated as such:** deep-context (150K+) interference isolation is an
  extrapolation; interrupt *rate* was never measurable with the counters used.

## The plan against reality

The plan document (`docs/training-plan.md`) is itself part of the record: its Key Changes log
tracks every divergence between the paper plan and what the work demanded. The shape of it: 24
weeks planned; +3 weeks inserted by reality (the Gemma 4 day-1 arc, its continuation, and the
parallelism closing chapter); +1 for a close-out week; then concluded at **16** when the capstone
was achieved in substance ahead of the paper plan. The original Phases 4–6 (Weeks 17–28) were
dispositioned: multi-model routing, the observability stack, and the "enterprise inference
platform" capstone were achieved in substance by the two-tier stack; RAG, application-side
routing, hardening, and the broad quantization quality-degradation curve migrated to a successor
program (`ai-engineering-training`, defined in its own plan — this document makes no claims about
it); speculative decoding, NSight profiling, and the cost and capacity-planning frameworks are
deferred as topics of continued interest.

## Model index

Every model referenced in this document, with the variants the program ran and where each appears
above:

| Model | Variants in the record | Role in the program | Where above |
|---|---|---|---|
| Llama 3.2 3B | FP32 / FP16 (transformers); FP16 (vLLM) | the Phase 1–2 baseline and framework-comparison model | Weeks 1, 3, 4, 5 |
| Llama 3.2 1B | FP16 (PyTorch); ONNX + TensorRT conversion | the Week 2 conversion-pipeline test subject | Week 2 |
| Llama 3.1 8B | FP16 | the Week 3 pipeline-parallelism test subject | Week 3 |
| Mistral 7B | FP16 | the single-GPU larger-model reference | Weeks 5–6 |
| Qwen 2.5 14B | FP16 | the tensor-parallel scale-up, and the subject of the Week 6→7 interconnect correction | Weeks 6–7 |
| Gemma 4 31B Dense | Q8_0 (llama.cpp, Week 8); FP8 (vLLM, Week 11); **QAT W4A16 (production orchestrator, Week 13 on)**; BF16 (its unquantized parent, as the Week 13 quality reference) | the orchestrator tier | Weeks 8, 11, 13–15 |
| Gemma 4 26B-A4B MoE | Q8_0 and Q4_K_M (llama.cpp); AWQ-INT4 (vLLM) | the day-1 deployment arc and the KV-sizing investigation vehicle | Weeks 8–10 |
| Gemma 4 12B | **QAT W4A16 (production workers, Week 12 on)**; BF16 (its unquantized parent, as the Week 13 quality reference) | the worker tier | Weeks 12–15 |
| all-MiniLM-L6-v2 | ONNX | the Week 6 Triton embedding deployment (the program's one non-LLM serving exercise) | Week 6 |

## Reading the record

| To understand… | Read |
|---|---|
| The architecture and its validation | [`docs/delegation-architecture.md`](./delegation-architecture.md) |
| Any single week | `week-NN-summary-*.md` in its phase/week directory (Weeks 8–15), or the `week-NN.md` report (Weeks 1–7) |
| A phase at a glance | `phase-1-summary-*.md`, `phase-2-summary-*.md` in the phase directories (Phase 3's summary follows at plan closure) |
| A specific measurement | the daily journal named in the weekly summary, and the result JSONs committed beside it |
| The plan's evolution | `docs/training-plan.md` §Key Changes |
| The published articles | `docs/linkedin/README.md` |
| This document's claim-by-claim sources | `phase-3-optimization-and-quantization/week-16/week-16-session3-capstone-claim-inventory.md` |

The serving conventions, endpoint footguns, and session discipline that operate the stack are in
this repo's `CLAUDE.md` and the toolchain repo's README — live documents, updated to current
truth, unlike the journals, which are the permanent record of what was believed and measured at
the time.
