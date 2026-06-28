# Week 14 — Session 3: Split completion and doc-sync

**Phase 3 — Optimization & Quantization**
**Date:** 2026-06-28 (UTC)
**Box:** `inference` — 4×RTX 3090; NVLink pair on GPUs 0+2.
**Image:** `vllm/vllm-openai:v0.23.0` (`sha256:6d8429e3…22ed8f`)
**Scope:** cleanup + docs only — no new experiments. Finish the Week-14-S1 repo split (R=data,
T=toolchain+eval-inputs), sync the live reference docs to the post-split reality, clear file-hygiene
drift.

Session discipline: inspect before deciding (no pre-assumption from the pickup's partial listing);
one logical change per commit; live-smoke each tool edit before committing; both repos committed
clean at the end.

---

## Opening triage

The S1 split left ambiguous state. Resolved, in order:

1. **Leftover `tools/` in R** — contained only a gitignored `__pycache__/`; `git ls-files tools/`
   returned nothing. The split had already `git rm`'d every source file when it moved to T. Plain
   `rm -rf` (no git involvement). R now has no `tools/` dir, so its README can truthfully point at T.

2. **Stray root JSON** (`exp_quality_rca_gemma-4-31B-it-qat-w4a16-ct_20260624T025121Z.json`) —
   untracked. Inspection showed a full, provenance-clean capture (`tool_git_dirty=False`, SHA
   `016485d8`, 8 probes), dated 2026-06-24 — S1/S2 window. No committed week-14 results sibling
   exists; the committed 31B-QAT captures all live under `week-13/results/` dated 06-16/17. Verdict:
   S2 live-validation **gate exhaust** (a pass/fail plumbing check, not an archival capture), landed
   in root because the run was cwd=R-root. Plain `rm`.

3. **venv recipe** — R keeps its `setup.sh`/`requirements.txt` (they correctly provision R's
   Phase-1 box stack: torch/tensorrt/onnx + R's directory scaffolding). T gets a **fresh slim
   recipe**, not a copy: T's tools are HTTP clients, so the only third-party import across the whole
   toolchain is `httpx` (verified by grepping imports — everything else is stdlib; the judge
   hand-rolls the Anthropic call over httpx, no SDK). A verbatim move of R's `setup.sh` would have
   pinned the wrong (heavy) environment and scattered R's directory tree into T.

---

## Tool changes (T) — each its own commit + live smoke

A live 12B-QAT worker (`gemma4-12b-smoke`, GPU 1, port 8000) was brought up to smoke against, then
later replaced with the 31B orchestrator for the MML confirmation (below).

1. **`vllm-bringup-checks.sh`: `--name` → `--container-name`.** No shell caller used `--name` (the
   only cross-repo hits were doc/journal prose, which we don't rewrite), so `--name` was kept as a
   **deprecated alias** that still works but warns to stderr — nothing copy-pasted from a
   week-13/14 journal breaks. Smoke: full PASS with the new flag (placement GPU 1, served-model
   match, chat smoke `finish=stop`); alias run emitted the deprecation warning and still resolved
   the container.

2. **`start-vllm.sh` + `run-judge.sh`: explicit `--help`.** `vllm-bringup-checks.sh` already had
   `-h|--help` (the pickup's note that it didn't was stale). `start-vllm.sh` had none — added
   `usage()` + `-h|--help) usage; exit 0`. `run-judge.sh` forwarded `--help` to the Python judge,
   which can't even show its own help outside the venv (it imports httpx at module top) — added a
   wrapper-level intercept ahead of all key-file logic, so `run-judge.sh --help` works with no key
   and no venv. Both smoked (help exits 0, unknown-arg exits 2; run-judge help proven outside the
   venv).

3. **`start-vllm.sh`: role presets + image default.** Replaced the retired FP8 zero-arg default with
   named role presets — a redesign chosen over a silent default bump because a zero-arg launch that
   *does* something is inherently a footgun, while one that shows the roles can't serve the wrong
   thing:

   | role         | model                              | mode | gpus | port | MML    | util |
   |--------------|------------------------------------|------|------|------|--------|------|
   | orchestrator | google/gemma-4-31B-it-qat-w4a16-ct  | TP=2 | 0,2  | 8000 | 131072 | 0.95 |
   | worker1      | google/gemma-4-12B-it-qat-w4a16-ct  | TP=1 | 1    | 8001 | 131072 | 0.90 |
   | worker2      | google/gemma-4-12B-it-qat-w4a16-ct  | TP=1 | 3    | 8002 | 131072 | 0.90 |

   Role is a positional arg consumed before the flag loop, so explicit flags still override. No role
   and no `--model` → usage + exit (no silent launch). IMAGE default bumped `v0.21.0` → `v0.23.0`.
   The full manual path (`--model … --mode pp --size 4 --device-order …`) is preserved for the
   PP/device-order experiments that fit no role. Worker TP=1 is not a free choice: GPUs 1/3 are on
   PCIe x1 with no NVLink, so TP=2 across them would let the all-reduce dominate decode — single-card
   workers keep the slow link off the critical path. Smoked all three role intent-echoes + the
   no-role and flag-override safety properties.

4. **`throughput_sweep.py`: `--parallelism` provenance tag in the default filename.** TP=1 vs TP=2
   result files were distinguishable only by timestamp. Added `--parallelism`
   (`choices=tp1/tp2/pp2/pp4/na`, default `na`), mirroring the existing `--placement` mechanism:
   folded into the filename before the placement segment, recorded in `sweep_config`, echoed in the
   run header. Help text is explicit that it's an **asserted, not measured** tag — the script is a
   pure HTTP client with no view of the server's topology (confirm via nvidia-smi uuid-join). Smoke
   (a real minimal sweep against the live endpoint): tagged run wrote `…_c1_tp1_<ts>.json`; untagged
   run wrote `…_c1_<ts>.json` (segment omitted) — non-breaking for existing call sites.

---

## Orchestrator MML confirmation (live, not a sweep)

The role presets pin MML 131072 for **both** tiers. The 31B value rested on a Week-13 §B-2 journal
number; since we were baking it into a preset and the docs this session, we confirmed it on the
current v0.23.0 stack — a boot-log read, not a characterization run (so within the no-experiments
scope). Disambiguation first: the often-cited **33024 was the FP8-matched comparison** config, not
QAT production. The QAT §B-2 ceiling walk (util 0.95) had established:

| MML     | KV pool (tok) | max concurrency |
|--------:|--------------:|----------------:|
| 33,024  | 105,609       | 3.20×           |
| 65,536  | 151,308       | 2.31×           |
| 131,072 | 193,837       | 1.48×           |
| 262,144 | refused       | est. max 218,624 |

Live boot of the 31B-QAT orchestrator at MML 131072 / util 0.95 reproduced this **to the token**:
KV pool **193,837**, max concurrency **1.48×**, available KV ~10.29 GiB. No drift from the journal —
the preset value is empirically vindicated on the current stack. (The boot log still recommends util
0.9907 to fully recover the CUDA-graph tax; that recovery was found non-viable on this box in Week 11
and we correctly leave util at 0.95.)

---

## Doc-sync

- **T `requirements.txt` + `setup.sh` + `README.md`** (committed; recipe pair grouped per the
  R/T-consistency intent). `requirements.txt` is `httpx>=0.27,<1.0` (compatible-release floor, not an
  exact pin). `setup.sh` mirrors R's skeleton minus the Phase-1 weight, targets the shared
  `~/ai-inference` venv, reuses-not-clobbers. README documents the split, the provenance model
  (`tool_provenance()` records T's SHA via `__file__`; `resolve_input()` CWD-then-tool-repo; outputs
  resolve against CWD into R), the run-from-R convention, the judge key mechanism, role presets, and
  the full eight-tool inventory (corrected to include `interference_probe.py` and `start-stack.sh`,
  which the first draft missed).

- **R `README.md`** rewritten to a **static** repo description — purpose (AI inference
  infrastructure / LLM serving), hardware, the R/T split + run convention, Quick Start, a single
  LinkedIn link, license. Deliberately **no** status/phase-map/findings (those live in the
  training-plan and journals, so the README doesn't rot), and all career-transition framing removed.

- **R `CLAUDE.md`** rewritten as the operational doc: two-repo split, run convention, session
  discipline, and the per-tool endpoint footguns — notably the **`/v1` suffix differs per tool**
  (`throughput_sweep` omits it and appends `/v1/…`; `rca_quality_probe` includes it and appends
  `/chat/completions`; `vllm-bringup-checks` takes host/port). Chat-endpoint-only, concurrency
  one-integer-per-invocation, verify-placement-empirically, retired-pre-v0.23-workarounds. Phase-2
  range corrected to Weeks 5–10 (was wrongly drafted 5–8; checked against the training-plan).

- **`training-plan.md`: left unchanged.** Its `tools/` references are all either dated weekly
  records or the Week-14 entry describing the split itself — historical, not live path references to
  "fix." Rewriting them would falsely imply the split predated the weeks they document.

---

## File hygiene

- **`week-03/CLAUDE.md` → `week-03-notes.md`** (`git mv`, content untouched). A `diff` proved it is
  **not** a duplicate of the root CLAUDE.md — it's distinct Week-3 content (experiment table,
  findings, week-3 git workflow) that merely shared the reserved filename, and carried a now-stale
  "no NVLink" hardware note (period-accurate for Week 3, pre-bridge). Renaming frees the functional
  `CLAUDE.md` slot without rewriting the record.
- **`issue-39133-comment-26b-moe-reproduction.md` removed** (`git rm`). It was a ticket-draft
  derivative; the underlying 26B-MoE KV-sizing analysis is preserved in four week-09 journals
  (verified by grep before deleting).
- **`week-04-vllm` under `phase-2-production/`** — nominally a Phase-1 week per the plan, but left
  in place: the move would rewrite paths for cosmetic gain, with breakage risk.
- **Bare vs slugged week dirs** — left as-is. Going forward, new week dirs are **bare `week-NN`**
  (recorded as a durable convention); existing slugged dirs not renamed retroactively.

---

## State at close

- **T:** five tool-edit commits + recipe pair + README; tree clean; pushed.
- **R:** README + CLAUDE.md rewrites + two hygiene commits (week-03 rename, issue-39133 removal);
  tree clean; pushed. Inference box rebased onto the Mac-side commits.
- Both repos accurate to the finished split. The split is **complete**.

---

## Carried forward (for the S3 close-out / end-of-week re-plan — NOT started)

1. **`start-stack.sh` revisit** (own session): now that `start-vllm.sh` has role presets,
   `start-stack` could potentially call it per-role instead of carrying its own bring-up. It also
   boots nginx + does host-RSS capture + has a teardown verb — substantial, deserves its own session.
2. **Tool interface-consistency pass** (own session): the endpoint contract is inconsistent across
   tools (the `/v1`-suffix split above) — now documented but not fixed. Decide one convention and
   align; sweep for other CLI drift (flag names, `--help` coverage, results-dir conventions).
3. **Training-plan re-scope** (end-of-week discussion): scope this plan strictly to inference
   engineering, spin AI-engineering into a separate future plan. Key points: the split scopes the
   plan's GOAL/trajectory, it does **not** purge repo contents (the LLM-as-judge and copilot context
   stay — they were instrumentation for inference training). The operator copilot is a **real work
   project**, so it's the **guide**, not the training deliverable. Both plans target generic,
   transferable topics (inference: serving substrate; AI-eng: agent-harness frameworks as a category
   — LangChain/LlamaIndex/LangGraph/DSPy etc.). **Action:** define an explicit, generic terminal
   goal for the inference plan (a self-contained finish line, not "ship the copilot").

## Standing Phase-3 remainder (unchanged by this session)

- **IRS nginx `zone workers 64k;` fix** (own session, different repo): missing zone → per-worker
  least_conn counters; load-test concurrently after applying.
- **Week 15 — speculative decoding** on the 12B QAT worker; **Week 16 — NSight profiling**; then
  Phase-3 close-out.
