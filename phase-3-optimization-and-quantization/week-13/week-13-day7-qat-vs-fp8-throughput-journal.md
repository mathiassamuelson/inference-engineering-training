# Week 13 Day 7 — QAT vs FP8 throughput (§B-(4) decode/prefill)

**Date (UTC):** 2026-06-17
**Session goal:** Close §B-(4) — QAT vs FP8 generation throughput at matched context, same
sweep ladder. Confirming this lands **win #2 (decode/prefill speed)** and, together with the
already-banked §B-1/2/3, fully unblocks the architecture write-up.
**Verdict:** **GO — §B-(4) closed, win #2 confirmed.** All three §B wins now land.

---

## Pre-session gate (all clean)

- **Git, both repos:** `git pull --ff-only` → already up to date, clean trees. Training HEAD
  `210d2a2` (Day-6 journal) with the FP8 results below it (`ca9b320`) — Day-6 push confirmed
  landed. IRS HEAD `f2a7af3` (nginx `/v1/` namespace move) — clean.
- **Container/GPU ground truth:** the two 12B workers (GPUs 1, 3, ~21.3 GiB each) survived from
  Day-6, up ~25h. **GPUs 0+2 cold (1 MiB) — orchestrator slot free, no teardown needed.**
- **Pinned digest verified by RepoDigests:** `vllm/vllm-openai:v0.23.0` @
  `sha256:6d8429e3…22ed8f`.
- venv `ai-inference`, `python3` explicit.

## Sequencing decision — single boot, no swap

Read the committed Day-4 FP8 anchor
(`throughput_sweep_…FP8-block_c1_20260614T194625Z.json`) before deciding. Its ladder:
prompt sizes `[512, 2048, 4096, 8192, 16384, 32768]`, concurrency 1, max_tokens 256,
iterations 3, warmup 1, `nonce_prefixed`. All six sizes sit under FP8's ~54,496 serviceable
ceiling. **The intended ladder matches the anchor exactly**, so FP8 is already banked — only
QAT is new. **One boot, zero swaps.** Comparison cells pulled from the committed FP8 JSON, not
memory.

Matched context enforced at the **config** level, not just prompt sizes: QAT booted at the
**same MML 33,024** the FP8 anchor used. QAT could serve far higher (Day-6: ~218K ceiling), but
raising MML changes KV-pool block-sizing and would break the comparison (the Day-3 "a rate
number is meaningless without its context length" trap).

## Boot + empirical placement verification

Launcher: `tools/start-vllm.sh --model google/gemma-4-31B-it-qat-w4a16-ct --mode tp --size 2
--gpus 0,2 --image vllm/vllm-openai:v0.23.0 --max-model-len 33024 --gpu-mem-util 0.95`.
Container named `vllm-tp2` (bare launcher names by image).

**Placement verified by UUID-join (not footprint):** container sees two GPUs —
- container idx0 → `GPU-3a7eac76…` → **host index 0** ✓
- container idx1 → `GPU-8b223d02…` → **host index 2** ✓

Host idx1/idx3 UUIDs (the workers) correctly absent from the container. vLLM bound to the
NVLink pair (0+2) as intended. `/v1/models` confirmed served id
`google/gemma-4-31B-it-qat-w4a16-ct` at `max_model_len: 33024`.

## `jit_monitor` reconciliation (record correction)

The Day-7 pickup referenced `jit_monitor` as "the Day-4 tool." **It is not a tool in our repo**
— `ls tools/`, `git log -- 'tools/*jit*'`, and `grep -ril jit tools/` all came back empty. It
is **vLLM's own built-in** (`jit_monitor.py` inside the worker), which surfaced in the container
logs during the sweep:

```
INFO  jit_monitor.py:54  Kernel JIT monitor activated — Triton JIT compilations during
                         inference will be logged as warnings.
```

The phantom-tool reference is corrected here so it stops propagating into future pickups. The
live JIT signal was in `docker logs` the whole time — no tool needed to be built. Contamination
guard this session was a `docker logs -f vllm-tp2 | grep -iE 'triton|compile|jit'` watch plus
post-hoc per-iteration spread.

## JIT compiles during sweep — occurred, absorbed by warmup

Two Triton kernels JIT-compiled at sweep start (timestamp 01:07:54, the 512 warmup/early
iterations):
- `_compute_slot_mapping_kernel`
- `kernel_unified_attention`

**Measured iterations are uncontaminated.** Per-iteration spread is <1% at every size with no
anomalous first iteration (512: 66.5/66.4/66.4; 32768: 47.3/47.3/47.3). The compiles landed in
warmup, which is exactly what warmup is for. Also seen at load: a `torch.compile` graph build
(compile range (1, 4096), 12.98 s; 27.34 s total) — load-time, not in measured path. The
heterogeneous-head-dim note forced `TRITON_ATTN` backend (expected for Gemma-4, prevents
mixed-backend numerical divergence).

## Result — matched-context comparison (c=1, max_tokens=256, median of 3 iters)

QAT from this session (`…qat-w4a16-ct_c1_20260617T010754Z.json`, git `210d2a2`, clean).
FP8 from the committed Day-4 anchor (`…FP8-block_c1_20260614T194625Z.json`, git `e4f0a627`,
clean).

| prompt | QAT decode | FP8 decode | decode Δ | QAT prefill | FP8 prefill | prefill Δ |
|-------:|-----------:|-----------:|---------:|------------:|------------:|----------:|
| 512    | 66.4       | 44.3       | +50%     | 2029        | 1953        | +3.9%     |
| 2048   | 64.6       | 43.5       | +49%     | 1932        | 1872        | +3.2%     |
| 4096   | 62.0       | 42.3       | +47%     | 1855        | 1797        | +3.2%     |
| 8192   | 57.7       | 40.2       | +44%     | 1743        | 1689        | +3.2%     |
| 16384  | 53.5       | 38.1       | +40%     | 1567        | 1524        | +2.8%     |
| 32768  | 47.3       | 34.8       | +36%     | 1315        | 1292        | +1.8%     |

(decode/prefill in tok/s)

## Reading — hypothesis held in sign AND slope

**Decode: QAT +36% to +50%, margin largest short-context, narrowing with depth.** Mechanism
exactly as predicted: decode is bandwidth-bound (weight bytes moved every token); w4a16 moves
half of FP8's weight bytes, so where decode dominates (short ctx) the win is largest. As context
grows, KV-cache traffic increasingly shares the bandwidth budget, diluting the weight-byte
advantage — hence the narrowing. The mechanism predicted not just the sign but the downward
slope of the margin.

**Prefill: QAT +1.8% to +3.9%, also narrowing.** Predicted "closer, compute-bound, both Marlin"
— confirmed. Prefill is compute-bound, so halving weight bytes barely helps; both run the same
Marlin INT4 compute path. The small edge shrinks at depth as context-dependent attention compute
grows relative to the weight-load component.

**Separation is total:** QAT's *worst* decode (47.3 @ 32K) beats FP8's *best* decode (44.3 @
512). No context length in this ladder has FP8 decode competitive.

## Provenance caveat (honest gap, not papered over)

The FP8 anchor was run Day-4 under a **different clean git SHA** (`e4f0a627`) and its JSON
**lacks an `image_digest` field**, so I cannot byte-verify both sweeps ran the identical
container. They ran the **same pinned tag** (`v0.23.0`) and **same MML** (33,024), and the
comparison is decode/prefill **rate at matched context** — robust to that gap. But I do not
claim digest-identical provenance I can't show. (Aside: the current `throughput_sweep.py` has no
`--container` flag and no container/digest-lookup path, so the pickup's "sweep container lookup
must match `vllm-tp2`" concern is moot for this tool version — there is nothing to match.)

## §B status after Day 7

All three §B wins land:
- **§B-2 headroom** (confirmed Day-6): KV pool grows with MML; ~218K vs FP8's ~54K ceiling.
- **§B-3 quant consistency** (confirmed Day-6): QAT near-lossless vs FP8 on RCA probes.
- **§B-4 speed** (confirmed today): QAT decode +36–50%, prefill +1.8–3.9% at matched context.

**Architecture write-up is unblocked.** Content = §A native full-stack snapshot + the three §B
wins (headroom, speed, quant consistency). **Hold the load-balance claim** until the IRS nginx
`zone workers 64k;` fix lands and a re-probe shows distribution across both workers.

## Artifacts this session

- Result: `…/week-13/results/throughput_sweep_…qat-w4a16-ct_c1_20260617T010754Z.json`
  (git `210d2a2`, clean, 6 rows) — written by the sweep.
- This journal.
- Day-8 pickup.

No tooling changed this session; clean tree maintained throughout. The result file embeds the
clean SHA. Next session: architecture write-up unblock OR the Day-8 QAT-vs-BF16 quality
characterization (separate session, own gate-zero).
