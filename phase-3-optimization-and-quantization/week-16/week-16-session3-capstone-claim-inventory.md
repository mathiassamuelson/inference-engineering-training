# Week 16 Session 3 — Capstone claim inventory

**Purpose:** the claim→source map for the capstone summary (target: `docs/program-capstone.md`,
filename to confirm). Every claim the capstone will make appears here with a register tag and a
meaningful source reference. The capstone's final text strips these tags; this committed
inventory is the durable trace. Post-commit changes are appended corrections, never rewrites;
the capstone's final text governs.

**Tags:** the results register — `measured` / `interpreted` / `assumed` / `not-measured-here` /
`open` / `unsupported-by-the-record` — plus `record` for non-measurement facts of the program
record (decisions, deliverables, publications, plan changes) traceable to a journal or live doc,
and `miss` for prediction misses preserved as such.

**Reference forms:** `<file> §<section>`. Weekly summaries cited as `wNN-summary`; phase
summaries as `p1-summary` / `p2-summary`; `plan` = `docs/training-plan.md`;
`arch` = `docs/delegation-architecture.md`; `linkedin` = `docs/linkedin/README.md`.

---

## A. Program frame

| # | Tag | Claim | Source |
|---|---|---|---|
| A1 | record | Program ran 2026-01-13 → July 2026; originally planned at 24 weeks, extended to 27 then 28, concluded at 16 with Phases 4–6 dispositioned (achieved-in-substance / migrated to the successor / deferred) | plan §Key Changes (rows 24→27→28→16) + §Disposition + footer |
| A2 | record | Phase boundaries: Phase 1 = Weeks 1–4, Phase 2 = Weeks 5–10, Phase 3 = Weeks 11–16 | plan §phase headings |
| A3 | measured/record | Hardware: 4× RTX 3090 (2 cards Week 1, 4 from Week 2), Gigabyte B650 Eagle AX, Ubuntu 24.04, CUDA 12.6; AORUS NVLink bridge from Week 7 on GPUs 0↔2 (NV4, ~100 GB/s); GPUs 1/3 on PCIe 3.0 x1 | p1-summary header; week-07.md §Context; plan footer |
| A4 | record | Week 4 is a Phase-1 week whose journal lives in `phase-2-production/week-04-vllm/` — a filing artifact deliberately left by the Week 14 reorg | p1-summary header; w14-summary §Session 3 |
| A5 | record | Three repos: R `inference-engineering-training` (results/journals/captures), T `ai-training-tools` (toolchain + eval inputs; SHA pins code+inputs), IRS `inference-reference-stack` (deployment stack) | R `CLAUDE.md`; w14-summary §Session 1; w10-summary |
| A6 | record | Week 16 Session 1 renamed the repos from `rtx3090-ai-training` / `rtx3090-ai-training-tools` — identity anchored to the work, not the hardware; GitHub redirects keep historical links live; old names never reused | plan §Key Changes (renames row); `week-16-session1-repo-renames-journal.md` |

## B. Phase 1 — Foundation & Baselines (Weeks 1–4)

| # | Tag | Claim | Source |
|---|---|---|---|
| B1 | measured | Transformers batch-scaling plateau ~5,000 tok/s on Llama 3.2 3B; per-sample collapse 84 → 4.2 tok/s (95%); the program's touchstone number | p1-summary §Week 1; week-01.md §Experiment 2 |
| B2 | measured | FP16 gave 1.56× (not 2–3×): decode is memory-bandwidth-bound, 504/936 GB/s (54% of peak) | p1-summary §Week 1; week-01.md §Experiment 1 |
| B3 | measured | Linear memory model: 6.47 GB + 13.03 MB × batch, R² 0.9999, max error 1.9% | p1-summary §Week 1; week-01.md §Experiment 3 |
| B4 | measured | Generic ONNX/TensorRT pipeline made LLM inference slower: 0.44× PyTorch on Llama 1B (weights left on CPU); 3B export OOM-killed | p1-summary §Week 2; week-02.md §Experiment 2 |
| B5 | measured | Topology: GPU 0 on x16, GPUs 1–3 on x1 (~1 GB/s), no P2P anywhere; ring all-reduce 378.9 ms for 32 MB → TP unviable on this pre-NVLink topology | p1-summary §Week 3; week-03.md §Topology |
| B6 | measured | Data parallelism 93.6% efficiency at 4 GPUs (7,422 tok/s); pipeline parallelism costs 8–18% (sync/bubbles, not bandwidth); PCIe x1 has no measurable effect on resident single-GPU inference | p1-summary §Week 3; week-03.md §Exp 2–3 |
| B7 | measured | vLLM single-GPU: ~1.3× over transformers at every concurrency; plateau moved 5,000 → ~6,100 tok/s, same shape (bandwidth is the wall); continuous batching = 95% latency cut for short requests in mixed traffic; PagedAttention 2.21× memory efficiency; zero failures at 1,200 concurrent requests | p1-summary §Week 4; week-04.md §Exp 2–4 |
| B8 | measured | SLA-constrained capacity: memory math says 1,200 users/GPU, throughput says 100–150, p95<2s SLA says ~25 — plan backward from the SLA | p1-summary §Week 4; week-04.md §Exp 3 |
| B9 | measured/record | In-phase supersession: Week 1's 344 KB/token KV figure assumed MHA; Llama 3.2 3B is GQA (8 KV heads) → 112 KB/token, matching vLLM's pool sizing; Week 1 not rewritten, Week 4 carries the correction | p1-summary §Supersessions; week-04.md §GQA Deep Dive |
| B10 | miss | Expectation misses preserved: vLLM would give 5–10× (got ~1.3×); TensorRT 1.5–2× (got 0.44×); PCIe x1 would hurt inference (it didn't, for resident serving) | p1-summary §Supersessions and misses |

## C. Phase 2 — Production Inference at Scale (Weeks 5–10)

| # | Tag | Claim | Source |
|---|---|---|---|
| C1 | measured | vLLM data parallel: 95.4% efficiency at 4 GPUs, 18,053 tok/s; sustained-load CV 0.023 | p2-summary §Weeks 5–7; week-05.md §Exp 1, 3 |
| C2 | measured | **7.12× system throughput over transformers** on identical hardware/workload, entirely from continuous batching (per-request latency 1.04× — near-identical) | p2-summary §Weeks 5–7; week-05.md §Exp 4 |
| C3 | measured (superseded) | Week 6: Qwen 2.5 14B TP=2 over PCIe: 316.5 tok/s peak, 19.69× cost/token penalty vs 7B — superseded by Week 7 as a measurement of the interconnect | p2-summary §Weeks 5–7; week-06.md §Exp 1–2; plan §Week 6 ("later revised") |
| C4 | measured | Week 7: same config on the NVLink pair: 3,018 tok/s peak, **9.53×**; latency CV 0.6%, p50→p99 spread 35 ms; bottleneck moved from communication to compute | p2-summary §Weeks 5–7; week-07.md §Exp 1–2 |
| C5 | record | Bridge verified on GPUs 0↔2, not 0↔1 as assumed — origin of the verify-placement-empirically rule | week-07.md §Context; p2-summary §Supersessions |
| C6 | interpreted | "The Week 6 finding was not a conclusion about 14B models; it was a conclusion about 14B models over PCIe x1" — the phase's methodological centerpiece, in the week's own words | week-07.md §Executive Summary / §Key Learnings |
| C7 | record | Week 8 pivot: curriculum paused to deploy Gemma 4 within 24 h of its 2026-04-02 release; a llama.cpp segfault (binary-searched to ~5,400 tokens) reported upstream and fixed overnight | w08-summary §31B arc; plan §Week 8 |
| C8 | measured | Corrected 31B Dense baseline: ~1,170 tok/s prefill plateau (uncorrected sweep 18% low), decode 23.9→20.4 tok/s over 519→28K | w08-summary §benchmark-artifact correction |
| C9 | measured/method | Server-side prefix caching silently corrupted prefill benchmarks in **both** engines (llama.cpp slot cache W8; vLLM prefix cache at 71.2% hit rate W9); fix = per-request nonce; rule = cross-check every benchmark against a server-side counter the script didn't compute | w08-summary §correction; w09-summary §Day 1 |
| C10 | measured + miss | NVLink→PCIe pair swap: dense prefill −21–30% vs ~1.5% predicted; `nvidia-smi topo -p2p r` shows CNS on all non-NVLink pairs → CUDA silently host-stages; sharpens Week 7 ("peer access existing at all") | w08-summary §topology + §prediction misses |
| C11 | measured | Dense-vs-MoE: 26B-A4B decodes 4.6–4.7× faster at every context length; fits full 262K (36.3 GB) where dense auto-shrank to 104K (45.7 GB) | w08-summary §head-to-head |
| C12 | interpreted | Per-cell KV cost (layers × KV heads), not parameter count, is the memory lever (~¼ per cell on the MoE) | w08-summary §head-to-head interpretations |
| C13 | measured/interpreted | Week 8 Day 4: six distinct failure modes ended the FP8-on-Ampere road; cleanest insight: FP8 *weights* (Marlin emulation, runs anywhere) and FP8 *KV cache* (needs SM 8.9+) have different hardware requirements; AWQ-INT4 via compressed-tensors served 262K at TP=2 | w08-summary §vLLM bring-up |
| C14 | record | `tools/throughput_sweep.py` created Week 9 — provenance metadata, self-describing filenames, nonce discipline, tokenizer calibration — the seed of the T-repo toolchain | w09-summary §Day 1 |
| C15 | measured (un-readjudicated) | Single-request TP-vs-layer-split crossovers (decode ~8K, prefill ~32K) — declared suspect the same week (#39133) and never re-adjudicated even after the fix; parked pending concurrent measurement | w09-summary §Day 2 + §Open |
| C16 | measured/record | #39133: KV pool flat at ~95,472 tokens across a 32× MML range; per-token arithmetic matched "all 30 layers sized full-MML" to 0.04%; contributed as a quantified reproduction to the existing upstream issue | w09-summary §Day 3 |
| C17 | record | **The pause:** Week 9 closed early rather than publish or measure concurrency on a known-buggy allocator — the program's defining discipline call | w09-summary §Day 3 close-out; plan §Week 9 |
| C18 | measured | HMA re-test (vLLM 0.21.0, May 17): 9.3× KV pool capacity (95K → 891K tokens at MML 262K), single-request throughput unchanged — the bug was allocator bookkeeping, not attention math | w09-summary §Day 4 |
| C19 | open | Left open by Week 9: K=V unification on global layers (a clean 2× over-allocation) and a ~400 MiB fixed per-sequence overhead — identified, not root-caused | w09-summary §Open |
| C20 | record | Week 10: public `inference-reference-stack` born (Compose: vLLM/nginx/Prometheus/Grafana/DCGM); Triton serving layer dropped on a concrete version conflict (NGC image bundled vLLM 0.15.1), decision documented in-repo; digest pinning as reproducibility anchor | w10-summary; plan §Week 10 |
| C21 | record | The Triton deep-dive never happened: introduced W6 (embedding model, 3.5× dynamic batching — measured), displaced by the W8 pivot, dropped W10; recorded as an honest arc, not a completion | p2-summary §Supersessions; week-06.md §Exp 3 |
| C22 | not-measured-here | Phase 2 never measured concurrent load on the fixed build (engine "max concurrency" figures are capacity math, not load tests) nor true vLLM prefill rates (TTFT-based lower bounds) | p2-summary §Not measured; w09-summary §Not measured |

## D. Phase 3 — the delegation arc (Weeks 11–15; cited from weekly summaries per the agreed sequencing inversion)

| # | Tag | Claim | Source |
|---|---|---|---|
| D1 | measured | Week 11 KV cost model (31B FP8, TP=2): per-seq KV/GPU ≈ 1.97 GiB + 39.2 KiB/token, validated to ~1%, predicted every later ceiling within 1.5% | w11-summary §Day 2, §Principles |
| D2 | measured | Text-only serving drops the vision tower entirely: 15.85 vs ~16.93 GiB/GPU weights, raising KV 3.25 → 4.04 GiB/GPU | w11-summary §Day 2 |
| D3 | measured/interpreted | PP=2 non-viable: the 256K-vocab embedding/LM head shard under TP but land whole on PP end stages, starving KV; serviceable context ~12× smaller than TP=2 | w11-summary §Day 3 |
| D4 | measured | Ceilings: TP=2 KV-bound at 54,496 (util 0.95) / 66,848 (util 0.97, +22.7%); PP=4 architecture-bound at the full 262,144 — but ~15 tok/s decode with ~5-minute TTFT | w11-summary §Day 6 |
| D5 | measured | Under c=4 load, TP=2 beat PP=4 on aggregate throughput and fan-out completion at every prompt size — capacity and throughput are distinct ceilings | w11-summary §Days 4–5 |
| D6 | interpreted | **The reframe:** fit is not the bar, usable-for-the-task is; "neither single config serves the use case" (TP=2 interactive but context-limited; PP=4 long-context but not interactive) — the finding that motivated delegation | w11-summary §Day 6 + §What it adds up to; arch §design-time measurement |
| D7 | measured/record | Week 12: the 12B QAT loads and serves on a single 24 GB card (8.28 GiB weights); Day 1's OOM was self-inflicted (`--hf-overrides` shallow-replace) plus one genuine image bug patched via a 3-line upstream backport, retired at convergence | w12-summary §debugging arc |
| D8 | measured + record | Worker ceiling: full 262,144 architectural context fits at 2.16× concurrency; production MML pinned to 131,072 as the model's validation boundary, not a memory limit | w12-summary §Context ceiling + §production call |
| D9 | measured | Worker throughput: decode 69.6 @8K / 51.7 @64K / 46.2 @102K tok/s; batching 2.33× at 8K, **functionally serial at 64K+** — direct input to the front-door design (queueing ≈ batching at depth) | w12-summary §Day 3 |
| D10 | record/measured | Week 13: version convergence onto one pinned image `vllm/vllm-openai:v0.23.0` (digest `sha256:6d8429e3…`), all Week-12 scaffolding retired, zero per-model workarounds across all three production models | week-13-summary.md §What landed; R `CLAUDE.md` (live) |
| D11 | measured | **QAT W4A16 ≡ BF16 parent at both tiers:** orchestrator guardrail adherence 8/8 tie; workers pointwise 4.83–5.0 with 6/6 strict format conformance, both components — via matched-provenance captures, position-bias-controlled pairwise + pointwise judging, deterministic format checks separated from the LLM judge | week-13-summary.md §Quality characterization + §Method |
| D12 | measured | QAT decode beats FP8 by +36–50% across the ladder (prefill +1.8–3.9%); the BF16-vs-QAT story is footprint (~2.4–2.7× smaller weights) | week-13-summary.md §Throughput |
| D13 | record | Eval toolchain built Week 13: `rca_quality_judge.py` (both-orders bias control), `rca_quality_probe.py`, `worker_contract_check.py`, `vllm-bringup-checks.sh` | week-13-summary.md §Tooling |
| D14 | record | Week 14: R/T repo split — `tools/provenance.py` anchors to its own `__file__` so every tool records **T's** SHA from any cwd; results resolve against cwd into R; host identity opt-in only; proven offline then live | w14-summary §Session 1, §Session 2 |
| D15 | measured | 12B worker parallelism: TP=2 on the NVLink pair wins at every concurrency (+47% c=1 decode, +81% prefill, aggregate lead growing to +72%); PP=2 ≈ TP=1; explicitly an NVLink result, not free-everywhere | w14-summary §Session 2 |
| D16 | measured (null) | **nginx zone null result:** the hypothesized least_conn skew did not reproduce — even split with and without the zone; mechanism: tie-fallback to weighted round-robin; both prediction halves wrong; fix kept as documented-correct, zero-cost; zone effect *unobservable* on this symmetric topology | w14-summary §Session 4 |
| D17 | measured | Orchestrator MML confirmation: 31B-QAT at MML 131,072 / util 0.95 reproduces the Week 13 ceiling walk **to the token** — KV pool 193,837, max concurrency 1.48× (and again live in Week 15) | w14-summary §Session 3; w15-summary §Day 2 |
| D18 | measured | **Week 15 operational proof:** under sustained saturation of the other tiers, neither tier loses more than ~0.5% decode at any measured size (R2 ≤0.25%; R3 0.41/0.49/0.06%) — ~7× inside the pre-committed 3% bar; both tiers isolated | w15-summary §Day 2; arch §operational proof |
| D19 | measured | The blind falsifiable commit held: worker degradation at 49,152 (0.06%) < at 512 (0.41%), as the handoff-rate mechanism requires; all five committed predictions held, four refinements logged as refinements | w15-summary §Day 2 |
| D20 | measured | nginx pool split 680/663 (50.6/49.4) during the R2 flood — even distribution confirmed on the v0.23.0 boot path; attribution honesty: credited to least_conn on symmetric backends, **not** to the zone fix | w15-summary §Day 2; arch §even load distribution |
| D21 | measured + interpreted | Mechanism: host CPU contention, userspace-dominated (~85% user / ~15% system), loaded-not-exhausted (busy 1% → 45–50%, no thread pegged, RAM flat); the softirq→hardirq arc ends "unsupported and unnecessary," not "proven absent" (counters measure servicing time, not rate) | w15-summary §Day 2 + appendix; arch §mechanism |
| D22 | interpreted / not-measured-here | 49,152 is a large-context point (~25% of the 193,837 ceiling), **not** near-ceiling — and the error behind the label is told candidly: with half a dozen ceiling-adjacent figures in circulation by Week 15 (33,024 comparison MML; 54,496 / 66,848 FP8 ceilings; 131,072 production MML; 193,837 QAT pool; 262,144 architectural), the Day 1 design anchored "near-ceiling" to the wrong one — "~54K" matches Week 11's superseded FP8 ceiling (54,496), not the QAT stack actually under test (193,837; the recorded "off by ~3.5×" ≈ 193,837/54,496). *[Which number was grabbed is interpreted — the dailies record the magnitude of the error, not its source; W14 S3's need to disambiguate 33,024 shows number-confusion was a live hazard.]* The empirical boot-log read caught it (ground truth over preset); deep-context isolation (150K+) is extrapolated (safe by monotone decay), not measured | w15-summary §Day 2 + appendix; w11-summary §Day 6 (54,496); w14-summary §Session 3 (33,024 disambiguation); arch §scope constraints 1–2 |
| D23 | record | `docs/delegation-architecture.md` finalized Week 15 under the full derived-doc contract; one remembered claim (12B MML pin fixed upstream) deliberately withheld as unsupported-by-the-record | w15-summary §Day 3; arch (the doc itself) |
| D24 | interpreted | The architecture's end-to-end validation chain, as the architecture doc closes it: viable (W12), quality-lossless under quantization (W13), co-resident (W13), evenly fronted (W15), interference-isolated (W15); thesis substrate-neutral, the consumer-GPU system is the proof case, not the thesis | arch §Thesis, §Closing the arc |
| D25 | record | Production stack (frozen): 31B-QAT TP=2 GPUs 0+2 :8000; 2× 12B-QAT TP=1 GPUs 1/3 :8001/:8003; nginx :8080; MML 131,072; util 0.95/0.90 (asymmetry need-driven); placement verified per boot | arch §proof-case system; R `CLAUDE.md` (live) |
| D26 | record | The QAT migration lifted the design-time ceiling constraint; constraints 1 and 3 (context/token economics; concurrency at depth) carry the motivation on the production stack; the counterfactual ("same decision would have followed") is marked a judgment, not a derivation | arch §case for delegation + §QAT migration |

## E. The method (first-class outcome)

| # | Tag | Claim | Source |
|---|---|---|---|
| E1 | record | Predict-before-measure with committed predictions, scored after: the W11 cost model predicted every ceiling within 1.5%; W15 committed a full prediction table plus a falsifiable mechanism commit *before* any measurement and scored all five | w11-summary §Principles; w15-summary §Days 1–2 |
| E2 | record | One experiment per session/boot; one variable at a time; `--warmup` justified empirically (the W11 cold-start artifact) | w11-summary §Principles + §Day 4; R `CLAUDE.md` (live) |
| E3 | record | Honest nulls preserved by name: the nginx zone null (W14 S4 — both prediction halves wrong, corrected mechanism recorded) and the softirq/hardirq arc (W15 — wrong counter measured first, right counter measured second, disposition stated as "unsupported and unnecessary") | w14-summary §Session 4; w15-summary §Day 2 + appendix |
| E4 | record | Never-rewrite journals; corrections are appended (W15 Day 2's same-day resolution; the Session 2 journal's two appended corrections) | w15-summary §Day 2; `week-16-session2-…-journal.md` §corrections |
| E5 | record | Provenance discipline: pinned image digests (from W10 on), tool git SHA in every result (`tool_git_sha`, anchored to the tool repo via `__file__`), commit-before-run (the dirty-tree trap), self-describing result filenames | w10-summary; w14-summary §Session 1; w11-summary §Principles; R `CLAUDE.md` |
| E6 | record | Empirical verification over declared intent: GPU placement by UUID→PID join every boot; ports confirmed live (the :8002/:8003 drift would have faked an isolation result); ceilings walked, not trusted (W12's "characterize the real ceiling" resolution) | w14-summary §S2; w15-summary §Day 2; plan §Week 12 outcome |
| E7 | record | Upstream engagement: llama.cpp Gemma 4 segfault reported day-1 and fixed overnight; vllm#39133 two-architecture quantified reproduction contributed; publish-nothing-on-a-known-bug pause (W9) | w08-summary §31B arc; w09-summary §Day 3 |
| E8 | record | Four published LinkedIn Pulses: 31B day-1 report (2026-04-04), dense-vs-MoE (2026-04-06), LLM-as-judge quantization quality (2026-06-21), TP=2-on-one-card (2026-06-24) — the public paper trail as deliverable | linkedin (publication index) |
| E9 | record | The claim-inventory/register practice itself (tagged inventories before prose in derived docs; this document is an instance) — adopted for the W15 write-up and the W16 consolidation | w15-summary §Day 3; `week-16-session2-…-journal.md` |

## F. Scoping and openness (carried into the capstone verbatim in spirit)

| # | Tag | Claim | Source |
|---|---|---|---|
| F1 | interpreted (settled scope) | All 4×3090 numbers are a **terminal characterization of a retiring topology**; the host-side coupling mechanism does not exist in the same form on a single-die successor; values must not be trended forward — what transfers is thesis, method, and pattern | w15-summary §Day 1; arch §scope constraint 3 |
| F2 | open | 131K–262K worker context range: memory fits it, quality-unvalidated (carried from W12) | w12-summary §production call; arch §scope constraint 4 |
| F3 | open | K=V unification on global layers (~2× KV over-allocation) and the ~400 MiB fixed per-sequence overhead — identified from data, never root-caused | w09-summary §Open |
| F4 | open | The TP-vs-layer-split single-request crossovers were never re-adjudicated post-#39133; the asymmetric-backend zone probe was scoped but not run | w09-summary §Open; w14-summary §Session 4 |
| F5 | not-measured-here | Deep-context (150K+) interference isolation is extrapolation; interrupt *rate* was never measurable with the counters used | w15-summary §Not measured |
| F6 | record | Plan disposition: Phases 4–6 (Weeks 17–28) not executed — routing/observability/capstone-platform achieved in substance by the two-tier stack; RAG, application-side routing, hardening remainder, quality-degradation curve migrated to the successor program; speculative decoding, NSight, cost/capacity frameworks deferred | plan §Disposition + §Key Changes |
| F7 | record | The successor (`ai-engineering-training`) exists as a dispositioned destination in the plan; the capstone records this in one sentence and makes no claims about it | plan §What Follows |

## G. Synthesis claims the capstone itself makes (tagged here because they are the capstone's own framing)

| # | Tag | Claim | Source basis |
|---|---|---|---|
| G1 | interpreted | The arc statement: "from a ~5,000 tok/s transformers plateau on a 3B model to a quality-validated, concurrency-proven two-tier delegation architecture serving a 31B orchestrator and two 12B workers on the same class of hardware" — each leg individually traceable (B1; D11; D18; D10) | composition of B1, D10, D11, D18 |
| G2 | interpreted | The program's recurring shape — predict, measure, be instructively wrong, correct in the open — was present from Week 1 (Expected-vs-Measured tables) and formalized by Phase 3 (committed predictions, falsifiable commits) | p1-summary §What Phase 1 handed; E1 |
| G3 | interpreted | The three biggest single numbers in the program are framework (7.12×, W5), interconnect (9.53×, W7), and allocator correctness (9.3× KV capacity, W9 epilogue) — infrastructure multipliers, not model-level optimizations | C2, C4, C18 |
| G4 | interpreted | The paper trail (journals, committed predictions, appended corrections, pinned provenance) is itself a program deliverable, on par with the running system | E1–E9; plan §Week 16 ("public paper trail") |
| G5 | interpreted | **Task-fit over headline capability is a through-line of the program** — the bar is sufficiency for the use case, not biggest/best-scoring. Anchors in the record: the W6 decision framework ("smallest model that meets quality requirements"; week-06.md §Decision Framework); the W8 deployment-category frame and the prompt-over-fine-tuning/bigger-model decision (w08-summary §head-to-head, §statmon-ai decision); the W11 usable-for-the-task reframe (D6); the plan's W12 operating principle ("highest-fidelity model that gives an acceptable context window"; plan §Week 12); and the one measured instance — the 12B judged good enough *in absolute terms* for its focused extraction job (week-13-summary §quality). Stated as a through-line visible at these points, **not** dated to a moment of emergence (untraceable), and **not** as "smaller matched bigger" (never measured — no comparison against a larger model exists in the record) | composition of C11–C12, D6, D11; week-06.md; plan §Week 12 |

---

## Structure the capstone will follow (for review alongside the claims)

1. **What this is** — one-paragraph orientation for a reader arriving cold (A1–A6).
2. **The arc in brief** (G1, G3).
3. **Phase 1 — baselines** (B1–B10).
4. **Phase 2 — production scale, the interconnect correction, the frontier pivot** (C1–C22).
5. **Phase 3 — the delegation arc, week by week** (D1–D26), closing with a summary-and-pointer to `docs/delegation-architecture.md` — the capstone owns the story, that doc owns the architecture.
6. **The method** (E1–E9, G2, G4) — first-class section.
7. **Scope, honesty, and what remains open** (F1–F5).
8. **The plan against reality** (A1, F6, F7) — 24→28→16, dispositions, one successor sentence.
9. **Repo map and the renames** (A5, A6).

No career-transition framing anywhere. Markdown tables permitted (GitHub-rendered).

**Filename:** `docs/program-capstone.md` (confirmed at the inventory review).

---

## Correction — appended during draft review (same session)

*Appended per the never-rewrite convention; adds, does not edit. The capstone's final text governs.*

**B4, scope tightened.** The inventory row is accurate as written (the 0.44× measurement, its
device-placement cause, the 3B export OOM). During draft review, a legibility expansion of
this claim in the capstone briefly asserted a *general root cause* — that the ONNX/TensorRT
recipe fails LLMs because their shapes aren't fixed at compile time, unlike image classifiers
it "reliably" speeds up. Neither leg is in the record: the week never investigated why a
correctly device-placed engine would or wouldn't win (root cause of the slowness was pinned
to device placement, full stop; the dynamic-ops attribution in the journal applies only to
the *direct-export trace failure*), and no vision model was ever measured (the only
non-LLM datapoint is a 1M-parameter toy network at 1.17×). Ruling: the general explanation is
**unsupported-by-the-record as a root cause** and was removed from the draft. The capstone
now states the three record-supported failure causes (device placement for the measured
slowdown; the trace error the journal attributes to dynamic ops for the direct export; RAM
exhaustion for the 3B export) and explicitly marks the counterfactual — whether a correctly
placed TensorRT engine would have beaten PyTorch — as not investigated.
