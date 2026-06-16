# Week 13 — Day 5: Step-4 execution & first native full-stack boot (v0.23.0)

**Date:** 2026-06-15 (US Mountain)
**Host:** `inference`
**Scope completed:** §A (the gate) — retire the gemma4-unified scaffolding and boot the full
multi-tier stack natively on one image, as a unit, for the first time.
**Scope deferred:** §B (the 31B FP8→QAT migration) — carried to Day 6. §A is a clean stopping point.

---

## One-line summary

The two-tier serving layout — two 12B QAT workers (GPUs 1 and 3) plus the 31B TP=2 orchestrator
(NVLink pair, GPUs 0+2) — now boots natively on a single image (`vllm/vllm-openai:v0.23.0`) with
**no source patch and no `--hf-overrides` blob**. The temporary 12B launcher and its patch are
retired from the repo. Empirical placement is clean; all three backends serve coherent chat; the
nginx front door routes correctly. One genuine finding surfaced (least-connection balancing did
not spread traffic) and was diagnosed to a missing config directive in the IRS nginx — a parked
follow-up, not a stack defect.

---

## Starting state (pre-session)

- venv `ai-inference` active; both repos clean; all four GPUs cold.
- `vllm/vllm-openai:v0.23.0` digest matched the Day-4 anchor
  (`sha256:6d8429e3…22ed8f`).
- Three stale exited containers from four weeks ago (`irs-vllm` exit 1, `irs-grafana`,
  `irs-dcgm-exporter`) were removed so the fresh boot wouldn't collide on names. Running
  observability (`nginx-frontdoor`, `irs-node-exporter`, `irs-prometheus`) left up.
- The QAT 31B checkpoint (`google/gemma-4-31B-it-qat-w4a16-ct`) was already cached, so §B-(1)
  will skip the pull — but §B itself was deferred.

---

## §A-2 — Step-4 execution: retire the scaffolding

### What the scaffolding was

Two workarounds were needed in Week 12 to run the 12B QAT model under the old preview image:
a **source patch** (a 3-line backport mounted over one of the image's Python files) and a long
**`--hf-overrides` blob** (re-stating the model's entire quantization config to dodge two launch
bugs). Day 4 proved both are unnecessary on v0.23.0 — the 12B resolves its architecture and
selects the Marlin INT4 weight kernel on its own. Day 5 made that real in the repo.

### Changes

- **Deleted** `tools/start-12b-qat.sh` (the temporary launcher) and the two Week-12 patch files
  (`gemma4_unified.py` + its `.orig` pristine copy). A pre-deletion grep confirmed nothing *live*
  still referenced them — only the launcher referencing itself, and the new retirement comments.
  Journal and committed-result references to the old launcher were left untouched (they are a true
  record of what ran then; editing them would falsify history).
- **Rewired** `tools/start-stack.sh` so both tiers launch through the one native launcher
  (`tools/start-vllm.sh`):
  - workers: `--mode tp --size 1 --gpus <single id>` (single-GPU; the slow PCIe-x1 link on GPUs
    1/3 is irrelevant with no cross-GPU traffic)
  - orchestrator: `--mode tp --size 2 --gpus 0,2` (the NVLink pair)
  - the converged image is pinned in `start-stack.sh` and threaded to each launcher via `--image`,
    so the stack orchestrator owns the image choice (which is the right layer for "one image, both
    tiers"). `start-vllm.sh`'s own default stays at v0.21.0 — correctly, since it is the Week-11
    reproduction anchor.
- **Second, non-obvious change forced by the first:** the native launcher uses host networking
  (`--network host`, no published port), whereas the retired launcher published a port. The stack's
  container-lookup helper had been finding containers by *published port*, which returns nothing for
  host-network containers — every tier's `image_digest` would have recorded UNKNOWN. Switched the
  lookup to match on the deterministic per-tier container **name** instead, which is robust across
  both networking styles. Also added `image_requested` to the result JSON.

Linted with `bash -n` (clean) before handover. Committed as the Step-4 change before any boot
that writes a result.

---

## §A-3/4 — Native full-stack boot + verification

### Predictions (made before the boot)

| Quantity | Prediction | Reasoning |
|---|---|---|
| Worker per-GPU memory | ~21–22 GiB | 8.28 GiB weights + KV filling toward util 0.90 × 24 GiB |
| Orchestrator per-GPU memory | ~22.8 GiB | 15.85 GiB weights + KV toward util 0.95 × 24 GiB |
| Time-to-healthy per tier | low minutes | graph capture (no eager mode); 12B at MML 131072 slowest |
| Swap used | ~0 | staggered boot avoids three simultaneous host-side loads |
| Worker pool balance (8 concurrent) | ≈ 4/4 | least-connection over two equal workers |

### Measured (clean boot, after the re-boot described below)

Empirical placement — physical GPU resolved by UUID → compute-process PID → container cgroup,
**not** trusting launcher intent:

| Container | Physical GPU | GPU memory |
|---|---|---|
| `gemma4-12b-qat-gpu1` | 1 | 21,250 MiB |
| `gemma4-12b-qat-gpu3` | 3 | 21,250 MiB |
| `gemma4-31b-tp2` | 0 | 23,408 MiB |
| `gemma4-31b-tp2` | 2 | 23,408 MiB |

- Workers landed on GPUs 1 and 3 as intended; the 31B occupies **both** GPU 0 and GPU 2 (two TP
  ranks, one per card) — the NVLink pair confirmed empirically.
- Per-GPU memory matched prediction: workers ~20.7 GiB (a touch under the band; util 0.90 is a
  ceiling, actual pool came in just below), orchestrator ~22.9 GiB (right at the util-0.95 ceiling).
- Time-to-healthy: workers 91 s / 83 s, orchestrator 105 s — all low-minutes, as predicted.
- Host RAM stayed comfortable: ~58.5 GB available at start, ~42.2 GB steady; swap effectively zero
  (64 MB).
- Direct chat probe to each backend (ports 8001 / 8003 / 8000) returned coherent `pong` — confirms
  output is real, not the token-repetition gibberish raw completions would produce on Gemma-4.

Front-door (nginx, port 8080) verification:

- `/healthz` → ok.
- Named-instance routes pin correctly: `/v1/worker/1/…`→8001, `/v1/worker/2/…`→8003,
  `/v1/orchestrator/1/…`→8000 (read back from nginx's per-request upstream log). The config's
  trailing-slash strip + path re-add does what its comment claims.
- **Note for client config:** a plain `/v1/chat/completions` through 8080 only reaches the *worker
  pool* (the 12B tier). The 31B orchestrator is reachable **only** via its named route
  `/v1/orchestrator/1/…`. This is by design.

### The one prediction that missed: pool balance

Predicted ≈4/4 across the two workers; observed **7/1**, then **8/0** on a cleaner re-probe.

- The first probe (48-token generations) was a *measurement artifact*: the generations finished
  faster than the eight requests launched, so they barely overlapped and the balancer almost always
  saw two idle workers.
- The re-probe used 256-token generations so all eight genuinely stayed in flight — and still landed
  8/0. That ruled out the artifact and pointed at the front-door config itself.
- **Diagnosis (real cause):** the IRS `nginx.conf` runs `worker_processes auto` but the `upstream
  workers` block has **no `zone` directive**. Without a shared-memory zone, each nginx worker
  process keeps its *own private* load-balancing counters. With several nginx workers each handling
  some of the eight connections, every one cold-starts its least-connection state from zero and
  picks the first-listed backend (8001). The pool is **not** broken — both workers are live and
  routable (the named routes hit both; the first probe reached 8003 once) — it simply isn't sharing
  balance state.
- **Fix (one line, parked):** add `zone workers 64k;` to the upstream block so the active-connection
  counts are shared across nginx workers. This lives in the **IRS repo** and belongs in the parked
  nginx-focused moment (so `nginx -s reload` survives git's inode swap on the mounted config). Not
  done tonight — different repo, separate experiment.
- **Consequence for the architecture write-up:** the full-stack snapshot (native boot, clean
  placement, working named routing) is intact, but the claim *"least-connection balances across both
  workers"* must be held until the `zone` fix lands and a re-probe shows real distribution.

---

## Operational notes (things that went sideways and how they were handled)

### A dirty-tree boot, re-done

The *first* §A-3 boot ran while the Step-4 changes were staged but **not committed** (a missed
`git commit`). The launcher's dirty-tree guard fired and the result JSON recorded the *pre-rewire*
SHA — i.e. a result file claiming to come from code that predates the native boot it documents.
That is exactly the misleading-provenance case the commit-before-results rule exists to prevent.
Resolution: committed the rewire, deleted the dirty-boot artifacts, and re-ran cleanly with
`--teardown-first`. The committed snapshot (`…233700Z.json`) records a post-rewire SHA with no
dirty warning. Also folded the empirical placement join and a corrected (long-generation) balance
probe into that single re-boot, so the redo cost one boot, not three.

### Remote/local divergence on push

`git push` was rejected: the inference checkout had never pulled the Day-4 journal commit
(`f8e9401`, pushed earlier from the Mac), while inference now carried tonight's two commits the
remote lacked — so the histories diverged (not a simple "behind"). Inspected before acting: the
remote commit adds **only** a journal markdown file; tonight's commits touch tools + results —
**zero file overlap**, so a rebase is conflict-free. Resolution is to rebase the two inference
commits onto `origin/main` (keeps history linear, matches the single-`main` model) then push.

- **Caveat to record:** the rebase rewrites the two inference commit hashes. The committed result
  JSON records `git_sha: 70791092…` (HEAD at measurement, the rewire commit). After the rebase that
  commit gets a new hash, so `git show 70791092…` will not resolve post-rebase — *the code content
  the SHA was meant to pin is identical*, only the hash label changed. Noting it so a future reader
  isn't confused by the dangling SHA.
- **Root cause / habit fix:** the normal flow is edit-on-Mac → push → pull-on-inference, which is
  one-directional. Tonight legitimately needed inference-*side* commits (result JSONs require
  inference's clean SHA — they can't originate on the Mac), and those two directions collided. The
  cheap prevention is **`git pull` on inference at session start**, alongside the clean-tree check.
  This is being added to the Day-6 pre-session block.

---

## Decisions banked

- **Step 4 is done in fact and in the repo.** The gemma4-unified scaffolding (patch + override +
  temporary launcher) is retired; one image serves both tiers natively.
- **Container lookup matches by name, not published port** — required once both tiers went
  host-network; keeps the result JSON self-describing (digest populated).
- **least-connection balance is an IRS follow-up, not an open question** — cause known (missing
  `zone`), fix known (`zone workers 64k;`), deferred to the parked nginx moment.

---

## State at close

- Full native stack **up** on v0.23.0: two 12B workers (GPUs 1/3, ports 8001/8003) + 31B FP8 TP=2
  (GPUs 0+2, port 8000) + nginx front door (8080). All tiers verified.
- Training repo: Step-4 rewire + §A snapshot committed on inference; **pending the rebase-onto-Day-4
  push** to sync with remote. Tree clean.
- IRS repo: untouched tonight. Carries the parked `zone workers 64k;` fix.

---

## Carries to Day 6

1. **§B — 31B FP8 → QAT w4a16-ct migration** (the headline arc, predict-before-measure at each step):
   - (1) pull-fit `google/gemma-4-31B-it-qat-w4a16-ct` on the NVLink pair (TP=2, GPUs 0+2).
     **Prediction: ~8 GiB/GPU** (≈ half FP8's 15.85, anchored on the 12B's native 8.28 GiB).
   - (2) re-characterize the MML ceiling — 4-bit frees ~7–8 GiB/GPU for KV; FP8 ceiling was 54,496
     at util 0.95; QAT could plausibly reach 100K+ **but the pool grows with MML — measure, do not
     extrapolate linearly**.
   - (3) quality-validate vs FP8 on the real statmon-ai prompt (QAT is quantization-aware-trained, so
     expect near-lossless — confirm, don't assume).
   - (4) benchmark decode/prefill vs the Day-4 FP8 anchor at **matched context**; test the Ampere
     hypothesis (native INT4 w4a16 both smaller *and* faster than Marlin-emulated FP8).
   - §B mutates the orchestrator (tears down FP8 on 0+2, boots QAT in its place); the two 12B
     workers stay up. The §A result commit/push should land **before** mutating the orchestrator.
2. **Parked (IRS, independent):** `zone workers 64k;` in `nginx.conf` + re-probe balance; the
   nginx directory-mount so reload survives git's inode swap.
3. **Architecture write-up** unblocks once both tiers are confirmed on w4a16 (native, one image) —
   its content is the §A native full-stack snapshot plus the §B three wins (context headroom,
   decode/prefill speed, tier-wide quant consistency). Hold the balance claim pending the `zone` fix.
