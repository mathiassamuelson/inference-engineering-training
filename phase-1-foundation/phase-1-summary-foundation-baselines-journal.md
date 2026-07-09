# Phase 1 Summary — Foundation & Baselines (Weeks 1–4)

**Span:** 2026-01-13 → February 2026 (Week 1: Jan 13–19; Week 2: Jan 20–26; Weeks 3–4: February, undated beyond the month in their reports)
**Hardware arc:** 2× RTX 3090 (Week 1) → 4× RTX 3090 (Week 2 onward); Gigabyte B650 Eagle AX; Ubuntu 24.04, CUDA 12.6. No NVLink in this phase (the bridge was ordered during Week 3).
**Models:** Llama 3.2 3B Instruct (the phase's reference model), Llama 3.2 1B (Week 2), Llama 3.1 8B (Week 3)
**Sources:** `week-01.md`, `week-01-qa-summary.md`, `week-02.md`, `week-03.md` (+ appendix), `week-03-notes.md`, and `week-04.md` — the last living in `phase-2-production/week-04-vllm/` (a known filing artifact; the training plan places Week 4 in Phase 1, and the Week 14 reorg deliberately left the directory unmoved).

## TL;DR

Phase 1 built the program's measurement floor with vanilla PyTorch and then demolished, one by one, the naive expectations it started with. Week 1 produced the number the whole program would be measured against — **the transformers library's batch-scaling plateau at ~5,000 tok/s** on a 3B model — plus a near-perfect linear memory model. Week 2 showed that a generic ONNX/TensorRT pipeline makes LLM inference *slower* (0.44× PyTorch), not faster. Week 3 mapped the motherboard's real topology (one x16 slot, three x1, no peer-to-peer) and established data parallelism as the only viable multi-GPU strategy on it — 93.6% scaling efficiency at 4 GPUs — while quantifying pipeline parallelism's cost and ruling out tensor parallelism entirely at PCIe-x1 bandwidth. Week 4 closed the loop with vLLM: the expected 3–5× throughput liberation turned out to be **~1.3×** (the plateau is memory bandwidth, which no framework fixes), and the framework's real value showed up elsewhere — continuous batching (95% latency reduction for short requests in mixed traffic), PagedAttention (2.21× memory efficiency), zero failures at 1,200 concurrent requests, and the SLA-constrained capacity lesson (25 real-time users per GPU, not the 1,200 that memory math suggested). The phase's recurring shape — predict, measure, be wrong in an instructive direction — became the program's method before it had a name.

## Week 1 — the baseline and the plateau

On 2× RTX 3090 with Llama 3.2 3B under the transformers library (measured):

- **Single-request:** FP32 53.95 tok/s → FP16 84.08 tok/s — a 1.56× speedup, not the expected 2–3×, because decode is memory-bandwidth-bound (504 GB/s achieved, 54% of the 3090's 936 GB/s peak; tensor cores mostly idle).
- **Batch scaling:** total throughput plateaus at **~5,000 tok/s** (batch 1200: 4,998 tok/s vs ~100,800 if scaling were linear — 20× short); per-sample rate collapses 84 → 4.2 tok/s. The journal's root-cause list (GIL, kernel launches, no fusion) was recorded as hypotheses, not verified attributions.
- **Memory model:** `Peak = 6.47 GB + 13.03 MB × batch`, R² 0.9999, max error 1.9% — the phase's cleanest measured artifact.
- Dual-GPU auto device-map correctly kept the 6.43 GB model on one GPU.

Week 1's capacity and cost figures (100–150 users/GPU; cloud-vs-on-prem TCO) were planning estimates of their moment, and the phase itself revised them (see Week 4). A separate Q&A document (`week-01-qa-summary.md`) captured the concept-level understanding — the two-bottleneck framing (bandwidth vs software), KV-cache mechanics, context-window-vs-sequence-length — rather than results.

## Week 2 — generic TensorRT pipelines fail for LLMs

The 4-GPU rig arrived; the planned TensorRT win did not (measured):

- SimpleNet (1M params): 1.17× — fixed ~10 μs overhead dominates microsecond inference.
- Llama 3.2 3B ONNX export: **OOM-killed** despite 57 GB free RAM (tied-weight duplication); TensorRT-LLM blocked on a CUDA-13 requirement; only the 1B exported.
- Llama 3.2 1B end-to-end: PyTorch FP16 183 tok/s; ONNX+TensorRT **81 tok/s — 0.44×, slower than baseline**. Root cause: ONNX Runtime left the weights on CPU (0.01 GB GPU memory was the tell), copying ~1.2 GB across PCIe per forward pass; its own `Memcpy nodes` warning confirmed it.

The finding that mattered was the *why*: kernel fusion and layout optimization are worthless if device placement is wrong, and generic pipelines don't manage placement, KV cache, or autoregressive generation. This is the phase's first "specialized frameworks exist for a reason" datum, before vLLM provided the positive case.

## Week 3 — topology is destiny: data parallelism wins on this box

The motherboard investigation (measured) found the constraint the rest of the program lived with: **one CPU-direct x16 slot (GPU 0), three x1 slots (~1 GB/s each), no peer-to-peer between any pair**. Inter-GPU bandwidth averaged 1.09 GB/s; a 32 MB ring all-reduce took 378.9 ms.

Four experiments established the strategy ranking on this topology:

- **Data parallelism:** near-linear to 3 GPUs (97–99%); **93.6% efficiency at 4 GPUs, batch 32 — 7,422 tok/s system throughput**. The 4th-GPU degradation at batch=1 (66%) was isolated to Python GIL contention, not hardware — pure GPU compute ran at 99.8% 4-way efficiency.
- **PCIe x1 has no measurable effect on single-GPU inference** (83–84 tok/s on every card): once weights are resident, decode runs on-chip. The link matters for loading, not serving.
- **Pipeline parallelism costs 8–18%**, and the overhead is synchronization/bubbles (~1–2 ms per stage boundary), *not* bandwidth — per-token activations are only 8–64 KB. Larger batches make it worse. Verdict: a last resort for models that don't fit one GPU.
- **Tensor parallelism: unviable at this bandwidth** (the all-reduce arithmetic gives ~30 s/token overhead on an 80-layer model). The appendix records the plan already in motion: an NVLink bridge on order to make a 2-GPU TP pair testable later.
- CUDA-streams work supplied the mechanics: different GPUs are naturally asynchronous (99.8%); same-GPU streams can't overlap compute with compute on saturated SMs (11.6%); compute+transfer overlap is free via the copy engine; pinned memory is ~40% faster for transfers.

## Week 4 — vLLM: the expected revolution arrives as ~1.3×, and the real lessons land elsewhere

vLLM 0.13.0, single GPU, same model and workload shapes as Week 1 (measured):

- **Throughput:** consistent 1.2–1.4× over transformers at every concurrency; ceiling moves 5,000 → ~6,100 tok/s with the same plateau shape. The gain is kernel-level (CUDA graphs, Flash Attention, torch.compile); the plateau is hardware bandwidth, which batching software cannot repeal. The Week 1 expectation ("vLLM should deliver 50–60 tok/s per sample at batch 512", implicitly 5–10×) is preserved as a miss: measured was 12.0 tok/s per sample at 512 — better than transformers' 9.2, but no revolution.
- **Continuous batching:** in mixed traffic, short requests finished in 0.27 s alongside 500-token requests taking 5.35 s — a 95% latency reduction vs static batching. Its value is latency management, not aggregate throughput.
- **PagedAttention:** 109 KB/token vs transformers' 261 KB — a constant 2.21× concurrency advantage at every sequence length.
- **Operational resilience:** zero failures at 1,200 simultaneous requests and across a 200-user simulation — the capability transformers simply lacks.
- **SLA-constrained capacity:** under p95 < 2 s, ~25 concurrent users/GPU; under p95 < 5 s, ~150. Against Week 1's memory-only estimate (1,200+) and throughput estimate (100–150), the phase's capacity-planning lesson is: **work backward from the SLA.**

**The GQA correction (in-phase supersession, recorded as such):** Week 1's per-token KV figure (344 KB) had assumed multi-head attention; Llama 3.2 3B uses grouped-query attention (8 KV heads, not 24), so the correct theoretical cost is **112 KB/token** — which matches vLLM's observed pool sizing almost exactly. Week 1's *measured* 261 KB/token stands as a measurement of KV-plus-framework-overhead, not of KV cache. The Week 1 report was not rewritten; Week 4 carries the correction.

One instrumentation gap was left open at phase close: vLLM V1's metrics endpoint reported 0% KV utilization under the Week 4 scraper (a naming incompatibility) — live KV observability deferred.

## Supersessions and misses inside the phase

- **Week 1 KV-cache arithmetic → corrected by Week 4** (MHA assumption → GQA reality, 344 → 112 KB/token).
- **"vLLM will shatter the plateau" → measured ~1.3×**; the plateau is bandwidth, and it moved, not broke.
- **"TensorRT gives 1.5–2×" → measured 0.44×** through the generic pipeline; the projected 1.4–1.5× for large models was never reached in-phase because the pipeline itself was the obstacle.
- **"PCIe x1 will hurt inference" → no measurable effect** for resident single-GPU serving; the deficit is real only for transfers.
- **"Multi-GPU always helps" → refined:** it helps as replicas (data parallel), hurts as splits (pipeline), and is impossible as shards (tensor parallel) — *on this interconnect*.

## What Phase 1 handed the rest of the program

- **The touchstone number:** Week 1's ~5,000 tok/s transformers plateau is the baseline every later serving result is implicitly measured against.
- **Topology-first thinking:** characterize the interconnect before choosing a strategy. The x16/x1/no-P2P map, and the finding that x1 doesn't matter for single-card serving, are the seeds of the eventual box layout (TP pairs need the good interconnect; single-card workers can live on the x1 slots). Phase 1's "tensor parallelism unusable" verdict is correct *for the pre-NVLink topology it measured*; the bridge that arrived later changed the premise, not the finding.
- **The proto-method:** every week closed with an explicit Expected-vs-Measured table. Predict-before-measure was already the working habit here; later phases formalized it.
- **SLA-backward capacity planning** and the three-tier framing (memory says 1,200, throughput says 100–150, the SLA says 25).

## Register notes

- All throughput/memory tables above are **measured** values from the week reports; capacity/user counts and cost tables are **period planning estimates** derived from them.
- Week 1's batch-plateau root causes are **hypotheses as recorded** ("Root causes?" in the original) — the plateau itself is measured; its decomposition was not verified in-phase.
- Week 3's per-boundary overhead attribution (sync/bubbles vs bandwidth) is **interpreted** from the magnitude mismatch (1–2 ms observed vs ~30 ms bandwidth-predicted), supported by the small-activation arithmetic.
- The Week 3 appendix's NVLink figures (56–900 GB/s classes, projected all-reduce times) are **reference calculations**, not measurements on this box.
- Weeks 3 and 4 report month-level dates only; no finer dating exists in the record.
