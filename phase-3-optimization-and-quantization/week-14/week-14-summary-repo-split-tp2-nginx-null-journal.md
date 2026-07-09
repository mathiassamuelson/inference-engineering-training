# Week 14 Summary — The R/T repo split, 12B-QAT TP=2 wins, and an honest nginx null result

**Dates:** Session 1 undated in its journal (its R-side reorg commit is dated 2026-06-22); Session 2 2026-06-24; Session 3 2026-06-28; Session 4 2026-06-30
**Cadence note:** Week 14 is where the program formally moved from *days* to *sessions*.
**Image:** `vllm/vllm-openai:v0.23.0` (`sha256:6d8429e3…22ed8f`) throughout the live work
**Repos:** the split happened under the original names — `rtx3090-ai-training` (R) and `rtx3090-ai-training-tools` (T); the Week 16 renames to `inference-engineering-training` / `ai-training-tools` came later. IRS = `inference-reference-stack`.

## TL;DR

Week 14 was a close-out/loose-ends week that produced three durable outcomes. First, the **R/T repo split**: the eval toolchain and its bundled inputs moved to their own public repo, retiring the dirty-tree provenance friction structurally rather than managing it — every tool now records the *tool repo's* SHA from any working directory, writes results against CWD (into R), resolves bundled inputs from either repo, and keeps host identity out of public output. Second, the week's one experiment: **TP=2 on the NVLink pair beats TP=1 for a 12B model that fits on one card** — +47% single-stream decode, +81% prefill, and an aggregate advantage that *grows* with concurrency to +72% — while PP=2 on the same two cards buys approximately nothing. Third, an **honest null result**: the long-carried nginx `least_conn` shared-zone fix was applied and load-tested, and the skew it was hypothesized to fix does not exist on this topology; the session's own prediction was wrong on two counts, and the corrected mechanism is recorded. Around these, the split was completed (tool cleanup, role presets, doc-sync) and the live reference docs were brought to post-split truth.

## Session 1 — the split (R = data, T = toolchain + eval inputs)

**Why:** all through Week 13, results landing in a tracked directory dirtied the tree, so the `git_dirty` provenance flag fired on benign sibling files. Two workarounds had been explored and discarded; the structural fix is that tooling and data should not share a repo. Post-split, a capture writes into R, the SHA that pins what produced it is T's, and writing results never touches T.

**The crux — provenance must record T's SHA, never CWD.** Tools live in `T/tools/` but run with CWD = R. The fix is one shared module, `tools/provenance.py`, anchoring to its own `__file__`; every tool that imports it records T's SHA regardless of launch directory. Key names were made honest (`tool_git_sha`/`tool_git_dirty`, probe schema 2→3) with a legacy-key fallback so committed Week 13 captures still read. R's SHA is deliberately *not* recorded — R being dirty at capture time is now expected and irrelevant. The crux was proven offline mid-session (recorded `tool_git_sha` = T's HEAD, not R's) before the move was finalized.

The remaining migration was one coherent pass over three recurring defects: result paths anchored to the tool repo (now CWD), `socket.gethostname()` leaking into public JSONs (now opt-in `--host-label`, default omitted), and hardcoded environment values (un-hardcoded). `resolve_input()` (CWD-then-tool-repo, same `__file__` anchor) made bundled prompts/probes/rubrics resolve from R without `$T/…` paths — with judge inputs deliberately left defaultless because both are tier-specific. Hygiene: `start-stack.sh` was held out of T's first commit until sanitized so the un-scrubbed text never entered public history; T's history is a clean cut whose initial commit names R's SHA.

**Part B — R reorg:** 18 files `git mv`'d into the phase/week tree (weeks 01–07 reports, the week-08/09 day journals, week-10's journals into a new `week-10-observability/` dir); `docs/` reduced to cross-cutting material; stale README/CLAUDE pointers fixed; historical journals not rewritten. The consumption model settled as side-by-side checkouts (submodules would re-entangle; pip-install fights the edit-run loop), accepting that R doesn't pin T's commit — each result JSON carries that itself.

Post-migration offline smoke passed all three gates; a live capture and a real judged run were **honestly deferred** to a box-up session — which became Session 2's first task.

## Session 2 — live validation PASS, then the TP=2 experiment

**Task 1 closed Week 13's deferred gate.** The live path — capture → `resolve_input` (prompt + probes) → provenance (T's clean SHA recorded from inside R's dirty tree) → live pairwise judge (12 calls, position-bias controlled) → verdicts — worked end-to-end. The judge verdict (5 ties, 1 order-sensitive non-tie, same model both sides) reads as a reproduction signal, not a quality finding. Two tooling lessons were paid for on the way:

- **The system prompt and the probe set are separate tier selectors and both must match.** The builtin default probes are orchestrator-tier; the first worker capture silently mixed tiers and was re-captured.
- **The pairwise judge fails silently on mismatched probe IDs** — zero API calls, zero verdicts, exit 0, a suspiciously small output file. Check `probes_scored`/output size, never exit code alone.

**Task 2 — the parallelism sweep** (12B QAT, 8192-prompt/512-gen matching Week 12 Day 3; each config solo; placement UUID-joined on every bring-up). The prediction was recorded first, with its uncertainty stated honestly: TP=2 was expected to win both legs, prefill confidently (compute-bound, split matmul); decode's *magnitude* was the genuine unknown — a tug-of-war between halved per-card weight reads and the per-token all-reduce.

| Metric (tok/s) | TP=1 | TP=2 NVLink | PP=2 NVLink |
|---|---:|---:|---:|
| c=1 decode | 70.0 | **102.6** (+47%) | 72.2 |
| c=1 prefill | 2,480 | **4,500** (+81%) | 3,275 |
| c=8 aggregate | 112.4 | **193.3** (+72%) | 117.0 |

Findings as the daily framed them: the tug-of-war resolves in favor of the bandwidth saving, and the +47% *is* the answer to "how does the balance land on NVLink" (the unsplit costs — KV reads, all-reduce, launch floor — consume the rest of the theoretical 2×). The aggregate lead **grows** with concurrency (+61 → +67 → +72%), the strong-direction confirmation. PP=2 is a wash against a single card for decode-heavy serving, with one clean asymmetry: it *does* help c=1 prefill (a single 8K prompt fills the pipeline) — a tidy illustration that PP's mechanism needs depth while TP's helps every token. Ranking at this workload: **TP=2 ≫ PP=2 ≈ TP=1.** PP over the PCIe x1 pair was reasoned out of scope, not measured.

**Caveat carried verbatim:** this is an NVLink result, not a free-everywhere result — the win rests on a cheap all-reduce.

A LinkedIn Pulse post on the finding was published the same session. One methodology debt was logged and later paid: TP=1/TP=2 result files were distinguishable only by timestamp (fixed in Session 3 with the `--parallelism` filename tag).

## Session 3 — split completion and doc-sync (cleanup only, no experiments)

- **Ambiguous state resolved by inspection:** R's leftover `tools/` held only a gitignored `__pycache__` (plain `rm`); a stray root capture JSON was identified as Session 2 gate exhaust, not an archival result, and removed; T got a fresh slim environment recipe (`httpx` is the toolchain's only third-party import — verified by grep) rather than a copy of R's heavy Phase-1 stack.
- **Tool interface fixes, each committed with a live smoke:** `--container-name` (with a warning deprecated alias so journal-copied commands keep working), explicit `--help` across the shell tools, and `throughput_sweep.py --parallelism` — documented as an *asserted, not measured* tag, since the script is an HTTP client with no view of server topology.
- **`start-vllm.sh` role presets** (orchestrator / worker1 / worker2) replaced the retired FP8 zero-arg default; no role and no `--model` prints usage and exits — a zero-arg launch that *does* something is inherently a footgun. Explicit flags still override. Image default bumped to v0.23.0. (The preset table this session recorded worker2 on port 8002 — Session 4 caught that as drift against the IRS pool.)
- **Orchestrator MML confirmation (a boot-log read, within the no-experiments scope):** the 31B-QAT at MML 131,072 / util 0.95 reproduced Week 13's ceiling-walk numbers *to the token* — KV pool 193,837, max concurrency 1.48×. The preset value is empirically vindicated on the current stack; the oft-cited 33,024 is disambiguated as the FP8-matched comparison config, not QAT production.
- **Doc-sync:** T README documents the split, provenance model, run-from-R convention, and the full eight-tool inventory; R README rewritten static (no status/findings to rot; career-transition framing removed); R CLAUDE.md rewritten operational — including the per-tool `/v1`-suffix footgun table. `training-plan.md` deliberately untouched: its `tools/` mentions are historical records, and rewriting them would falsely imply the split predated the weeks they document.
- **Hygiene:** `week-03/CLAUDE.md` renamed to `week-03-notes.md` after a diff proved it was distinct period content, not a duplicate; the stray issue-39133 ticket-draft removed after grep-verifying its analysis survives in the week-09 journals; bare `week-NN` recorded as the go-forward directory convention with no retroactive renames.

## Session 4 — the nginx `least_conn` zone fix: a preserved null result

The long-carried hypothesis: the IRS `workers` upstream uses `least_conn` without a `zone`, so under `worker_processes auto` each nginx worker process balances on private counters, and traffic was expected to skew (serial traffic pinning to the first server; concurrency required to reveal balance).

**The skew did not reproduce.** With and without the zone, across fresh-connection and keepalive, serial and c=2/4/20, the two 12B workers split within noise (e.g. no-zone 67/73 at c=20; with-zone exact 60/60). Both halves of the prediction were wrong. The corrected mechanism, from the data: `least_conn` falls back to weighted round-robin on tied connection counts; short requests are tie-dominated, and the per-process round-robin pointer alternates evenly even with unshared state. The zone's effect is **unobservable on this topology** — a symmetric two-worker pool can't manufacture the sustained asymmetric-count divergence the zone defends against. Exposing it would require deliberately induced backend asymmetry: noted as a future probe, not run.

The fix was **kept anyway** (nginx's documented requirement for shared stateful-balancer state; zero cost; defensively correct for heterogeneous pools), and the config's inline comment was rewritten to match the measured reality instead of the refuted "pins to the first server" story.

Two measurement artifacts nearly manufactured a fake skew, both caught: a cold-start transient (26/14 that vanished on repetition) and `docker logs` persisting across `docker restart` (weeks of stale traffic in the first count — every capture re-scoped with `--since`). A third operational catch: a bind-mounted *file* keeps serving the old inode after an editor's atomic-rename write — verify with `nginx -T` (running-config dump), never with reload success.

## Not measured this week

- The zone directive's actual effect under asymmetric backends — declared unobservable on this topology; the asymmetric-backend probe was scoped as a future experiment and not run.
- PP=2 over the PCIe x1 pair (Session 2's Config 4) — reasoned strictly worse than the NVLink PP result and skipped.
- Session 2's judge verdict is pipeline validation, not a model-quality measurement — the captures differ in harness version, not model.

## Open at week close (as the sessions left them)

- **T↔IRS drift:** worker2 role preset says port 8002, the IRS nginx pool expects 8003 (worked around per-launch with `--port`); the `nginx-frontdoor` file bind-mount should become a directory mount. Both bundled into the pending start-stack review.
- **`start-stack.sh` revisit** (could delegate per-role to `start-vllm.sh`) and a **tool interface-consistency pass** (the `/v1`-suffix contract, flag drift) — each its own session.
- **IRS local `main` 2 commits ahead of origin, push/hold undecided** at Session 4 close.
- **Training-plan re-scope discussion:** scope the plan strictly to inference engineering with a generic terminal goal (the operator copilot is the guide, not the deliverable); AI-engineering spins into a separate future plan.
- Per-week dir-name consistency pass — deferred again, now superseded by the bare-`week-NN` go-forward convention.

## Methodology lessons logged this week

- **Fix structure, not symptoms:** two provenance workarounds were discarded unbuilt once the repo split removed the friction at the root.
- **Put a subtle invariant in exactly one module** (`provenance.py`'s `__file__` anchor) so it cannot drift between tools.
- **Silent-success failure modes need positive evidence:** the zero-call judge run exits 0; check `probes_scored`. Reload success is not proof the new config is live; `nginx -T` is.
- **Predict before measuring, and keep the wrong ones:** the decode-magnitude question was explicitly framed as unanswerable a priori (and answered: +47%); the nginx skew prediction was wrong twice over and the corrected mechanism is the deliverable.
- **Scope capture windows** (`docker logs --since`) and **repeat suspicious first measurements** (the cold-start transient).
- **Deprecate, don't break:** journal-copied commands keep working via warned aliases; asserted metadata (`--parallelism`) is labeled as asserted.

## Artifacts produced this week

- The T repo in its complete post-split form: `provenance.py` (+ `resolve_input`), eight migrated tools with CWD-results/host-label/schema fixes, role presets, slim recipe, README
- R reorg (18 relocations), static README, operational CLAUDE.md
- Sweep JSONs for TP=1 / TP=2 / PP=2 (12 runs; TP files distinguishable only via the session journal's timestamp table — the gap that motivated `--parallelism`)
- IRS commit `09bc82b` (zone fix + honest comment)
- Published Pulse post on the TP=2 finding
- Four session journals

## Carried into Week 15

Week 15 was planned at Session 3 close as speculative decoding on the 12B worker; the standing Phase-3 remainder also listed NSight profiling (Week 16) and the close-out. (What Week 15 actually became — the cross-tier interference arc — is its own summary.)
