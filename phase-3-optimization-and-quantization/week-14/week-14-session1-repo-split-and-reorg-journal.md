# Week 14 · Session 1 — Repo split (toolchain → `T`) and reorg of `R`

**Phase 3 · close-out / loose-ends week.** No GPU work this session (box idle, workers torn
down end of Week 13). This was repo maintenance: split the eval toolchain and its inputs out
of the data repo into their own public repo, make every tool correct and public-safe under the
split, and consolidate the journal/report layout in the data repo.

Cadence note: these are **sessions** now, not days.

## Repos

| Tag | Repo | Holds |
|-----|------|-------|
| `R` | `rtx3090-ai-training` | results, journals, captures (the data repo; runs are launched from here) |
| `T` | `rtx3090-ai-training-tools` | toolchain **+** eval inputs (`prompts/`, `probes/`, `rubrics/`) |

Both public. `T`'s SHA pins code **and** inputs together, by design.

## Why split

All through Week 13, results written into a tracked dir dirtied the tree, so the `git_dirty`
provenance flag fired on benign sibling result files. Two workarounds were explored and
discarded (auto-commit results; a path-classifier `dirty_tracked` flag with a provenance-policy
config). The real fix is structural: **tooling and data should not share a repo.** Once split, a
capture writes into `R`, the SHA that pins what produced it is `T`'s, and writing results never
touches `T`. The friction disappears rather than being managed. Neither workaround was rebuilt.

## Decisions settled at session start

1. **Consumption model → side-by-side checkouts.** `R`, `T` (and the IRS stack) are siblings,
   all pulled at session start. Gives the full payoff: `R`'s tree may be as dirty as it likes
   with landing results without touching `T`, so `T` stays clean and `T`'s SHA is what gets
   recorded. Submodule would re-entangle (R tracks a T pointer); pip-install would fight the
   edit→push→pull→run loop. Trade-off accepted: `R` does not pin the exact `T` commit it used;
   provenance lives in each result JSON's metadata instead — same fidelity, since result files
   are already self-describing.
2. **`R` weekly-report location → the phase/week tree** (see Part B).
3. **Public-hygiene scrub → verify-on-move** (see Part A, hygiene).
4. **History → clean cut.** `T` already held a commit with the toolchain (seeded for the crux
   proof), so a `filter-repo` rewrite would have meant resetting `T` and force-pushing for the
   marginal benefit of a mostly-noisy log. A clean initial commit naming `R`'s SHA preserves the
   provenance pointer without the log.

Already-decided going in (not reopened): eval inputs move to `T` with the toolchain; both repos
public.

---

## Part A — the split

### The crux: provenance must record `T`'s SHA, never cwd

Post-split the tools live in `T/tools/` but run with **cwd = `R`** (results are written into
`R`). A naive `git_provenance(cwd)` would record `R`'s SHA — relocating the problem, not solving
it. The load-bearing fix: a single shared module, `tools/provenance.py`, whose `tool_provenance()`
anchors to **its own `__file__`** (`Path(__file__).resolve().parent` → `T/tools/`, then
`git -C` walks up to `T`'s `.git`). Every tool that imports it records `T`'s SHA regardless of
where it is launched from.

Design points:

- **One home for the invariant.** The subtlety (anchor to the tool repo, never cwd) lives in
  exactly one place so it cannot drift between tools. Import is free under side-by-side: tools run
  as `python3 tools/<tool>.py`, so `tools/` is `sys.path[0]` and a sibling `from provenance import
  tool_provenance` resolves with no packaging.
- **Honest key names.** Standardized all tools on `tool_git_sha` / `tool_git_dirty` — the name
  states the fact (it's the tool repo's SHA). The probe previously emitted a generic `git_sha`;
  bumped its `SCHEMA_VERSION` 2 → 3. Consumers (judge, contract-check) read `tool_git_sha` with a
  fallback to legacy `git_sha`, so already-committed Week-13 captures (v≤2) still read.
- **We deliberately do not record `R`'s SHA.** `R` being dirty at capture time is now expected
  and irrelevant to what produced the result; recording it would re-introduce the noise the split
  removes. The `tool_git_dirty` flag is the discipline gate that matters now.

**Mid-session crux proof (offline, no serving):** ran `worker_contract_check` and a judge
`--dry-run` from cwd = `R` against committed captures. Recorded `tool_git_sha` = `fdea6f08…`
(`T`'s HEAD at the time), **not** `R`'s HEAD `c0b654ea…`. The capture's `capture_provenance`
surfaced the old `git_sha` via the legacy fallback. Crux confirmed before the move was even
finalized.

### Remaining-tools migration (one coherent pass)

The same defects appeared across the rest of the toolchain; fixed together as "make the remaining
tools correct and public-safe post-split":

| Tool | Provenance → `tool_provenance()` | Results → cwd (not the tool repo) | Hostname scrub | Other |
|------|:---:|:---:|:---:|-------|
| `rca_quality_probe.py` | ✓ (earlier) | n/a (explicit `--results-dir`) | n/a | `--system-prompt` default + `resolve_input` (below) |
| `rca_quality_judge.py` | ✓ (earlier) | already cwd | n/a | `resolve_input` on `--reference-prompt`/`--rubric-file` |
| `worker_contract_check.py` | ✓ (earlier) | already cwd | n/a | legacy-key fallback reader |
| `throughput_sweep.py` | ✓ (dropped the results-exclude dirty workaround) | ✓ (was `REPO_ROOT`=T via `anchor_path`) | ✓ `--host-label` | `SCHEMA_VERSION` 3 → 4 |
| `interference_probe.py` | ✓ | ✓ (default was `REPO_ROOT/…/week-13/results` = T) | ✓ `--host-label` | `nginx-frontdoor` un-hardcoded → `--nginx-container` (default), gated on a `captures_nginx` preset flag |
| `start-stack.sh` | already `T` via `BASH_SOURCE`; emitted key `git_sha` → `tool_git_sha` | ✓ (was `$ROOT/…` = T) | ✓ `--host-label` (replaced auto `os.uname().nodename`) | dropped the `DIRTY_PATHSPEC` exclude workaround |
| `run-judge.sh` | n/a (self-anchors to the judge) | n/a | n/a | usage doc refreshed for post-split input paths |

Recurring themes:

- **Results-dir footgun.** Several tools anchored relative result paths to the *tool* repo
  (`REPO_ROOT` / `$ROOT` / `anchor_path`). Post-split that writes into `T`, re-dirtying it — the
  exact thing the split removes. All now resolve results against **cwd** (you run from `R`),
  while provenance stays anchored to `T`. The two concerns were conflated pre-split (both keyed
  off the script's repo root); they are deliberately separate now.
- **Hostname leak into public output.** `socket.gethostname()` / `os.uname().nodename` were being
  written into result JSONs that land in public `R`. Replaced with an opt-in `--host-label`
  (default omits the field) — capture host identity explicitly when wanted, never auto-scrape.
- **No hardcoded environment values.** `interference_probe.py`'s `nginx-frontdoor` container name
  moved out of the preset data structure into `--nginx-container` (default `nginx-frontdoor`),
  used only when the direction floods the load-balancer pool.

### Bundled-input resolution (`resolve_input`)

The eval inputs ship in `T` alongside the tools, but the RCA tools required explicit input paths
— and a relative `prompts/…` no longer resolves from cwd = `R`. Added `resolve_input()` and
`tool_repo_root()` to the shared module (same `__file__` anchor as provenance). Resolution order:
absolute → as-is; relative → cwd if it exists there, else the tool repo root; if neither, returned
unchanged so the not-found error names what was asked.

- `rca_quality_probe.py`: `--system-prompt` is no longer `required`; defaults to the bundled
  orchestrator prompt (`prompts/operator-copilot-rca-system-prompt.md`), resolved via
  `resolve_input`. `--probes-file` resolves the same way.
- `rca_quality_judge.py`: `--reference-prompt` and `--rubric-file` resolve via `resolve_input`.
  Left **without** hard defaults on purpose — an absent reference and the builtin rubric are valid
  common modes, and both are tier-specific, so defaulting could silently inject the wrong one.

Net: `python3 tools/rca_quality_probe.py --model …` and
`run-judge.sh … --reference-prompt prompts/…` now work from `R` without spelling out `$T/…`.

### Public-hygiene scrub (verify-on-move)

- `start-stack.sh`: the recurring leak was `os.uname().nodename` written into every
  boot-choreography JSON → replaced with opt-in `--host-label`. One comment that named the
  internal IRS repo was genericized. The pinned image digest (`vllm/vllm-openai:v0.23.0`,
  `sha256:6d8429…`) is the **public** image digest — benign, kept.
- `start-vllm.sh`, `vllm-bringup-checks.sh`: reviewed line-by-line, published unchanged (no
  hostnames/IPs/credentials; GPU index defaults and `~/.cache/huggingface` mount are benign).
- Decided to keep the bringup shell scripts in public `T` rather than a public/private split — a
  private/public seam through one coherent toolchain creates cross-repo mess (prior experience).
- `T` `.gitignore` added (`__pycache__/`, `*.pyc`) before the first commit — a `.pyc` of
  `provenance.py` had been produced by the crux smoke run.

### Move mechanics (clean cut)

`start-stack.sh` was deliberately **held out** of `T`'s first commit so its un-sanitized text
(which named the IRS repo) never entered `T`'s history; the sanitized version was its first
appearance, via a Mac round-trip. Every moved path lives in exactly one repo at all times
(never zero). `R`-side removal: `git rm -r tools prompts probes rubrics`, with `start-stack.sh`
removed last once its sanitized twin was in `T`.

---

## Part B — reorg of `R`

**Decision: consolidate weekly reports/journals into the phase/week tree** (matching the active
Phase-3 convention; weeks 11–13 already live there). The alternative — pulling everything into
`docs/weekly-reports/` — would have moved the recent, active weeks *away* from their own results
and broken the live convention.

18 files relocated via `git mv` into the existing per-week dirs: weeks 01–07 reports, the eight
week-08/09 day-journals (the worst offenders — report in `docs/`, code in `phase-2/`), and
week-10's two journals into a **new** `phase-2-production/week-10-observability/` (week 10 had no
phase dir because its stack work lived in IRS). `docs/` now holds only cross-cutting material
(`linkedin/`, `training-plan.md`). Per-week dir *names* are still inconsistent
(`week-01-benchmarks` vs `week-06` vs `week-11`); the rename-to-consistent pass was **deferred**
to avoid ballooning scope and touching many references.

Stale references fixed: `README.md` (broken link to `week-01.md`) and `CLAUDE.md` (structural
pointer to the now-gone `docs/weekly-reports/`). The `training-plan.md` mention was left as-is —
it's prose inside this session's plan entry describing the starting state, accurate as a record.
Historical journals were **not** rewritten (they were accurate when written; only live pointers
in README/CLAUDE/plan get fixed).

---

## Post-migration smoke validation (the gate)

The open item from the Week-13 close: prove the tools survived the move. Run from cwd = `R`
against the fully-migrated `T` (`016485d`), **zero serving**:

| Check | What it exercises | Result |
|-------|-------------------|--------|
| `worker_contract_check` (offline, against a committed v≤2 capture) | `tool_provenance()` from cwd=R; `--results-dir` → cwd (`/tmp`); legacy-key fallback | `tool_git_sha = 016485d…` (`T`, not `R`'s HEAD), `dirty=False`; `capture_provenance.tool_git_sha = 04c74db…` via fallback; 6/6 conformant; wrote to `/tmp` |
| `rca_quality_judge --dry-run` (header only) | `resolve_input('prompts/…')` from cwd=R where no `prompts/` exists; provenance | `reference : yes` (bundled prompt found in `T`); `tool git_sha = 016485d…` |
| `rca_quality_probe --help` | new `--system-prompt` default + import wiring | default wired, help renders, no traceback |

The `reference : yes` line is the load-bearing one: it proves `resolve_input` finds the bundled
prompt **in `T`** while running from `R`, through the real split layout.

**Not covered (honestly deferred to a box-up session):** a live capture (probe hitting a served
model) and a real judged comparison (Anthropic API spend). Everything the *move* touched is
confirmed; the serving paths are unchanged code and will be exercised when the stack is next up.

---

## Commit trail

**`R`** — `db7f551` split out toolchain+inputs · `c37f1f9` remove `start-stack.sh` (sanitized
copy now in `T`) · `19e21a4` reorg + reference fixes. (Base `c0b654e`.)

**`T`** — `fdea6f0` toolchain (crux-proof seed) · `195a765` import toolchain+inputs @ `R c0b654e`
· `e1edae9` boot-choreography orchestrator (sanitized `start-stack.sh`) · `1966fcc` adapt to
split: tool-repo provenance, CWD results · `016485d` resolve bundled inputs via `resolve_input`.

Both machines verified at the same HEAD on each repo.

---

## Carry-forwards / open items

- **Live validation** (capture against a served model; real judged comparison) — needs the box
  up. The provenance/results/resolver paths are already proven offline.
- **Per-week dir-name consistency pass** — deferred; many references would move.
- **Minor drift, optional fold-in** — the stray `issue-39133-…-reproduction.md` inside a
  `results/` dir; the duplicate `CLAUDE.md` under `phase-1-…/week-03-…/`.
- **Untouched Week-14 items, each its own later session** — nginx load balancing (the missing
  `zone workers 64k;` in the IRS upstream block → per-worker private `least_conn` counters);
  throughput Pulse post; 12B-QAT parallelism sweep.

## Standing habits reinforced this session

- **`git pull --rebase` on `R` before pushing.** `R` now has two writers (inference host for
  results/captures, Mac for edits), so divergence is the norm, not the exception — hit twice this
  session. Sits alongside the existing `git pull --ff-only` pre-session gate.
- **zsh `#` is not an interactive comment** unless `setopt interactive_comments`. Pasting a
  comment block ran the lines as commands and a `… -> it` comment was read as `> git`, creating an
  empty `git` file at the repo root (removed before commit). Don't paste comment blocks into the
  interactive shell.

## Headline

The dirty-by-sibling provenance friction is retired at the root, not managed. A capture writes
into `R`; the SHA that pins what produced it is `T`'s; writing results never touches `T`. Every
tool records the tool repo's SHA from any cwd, resolves its bundled inputs from any cwd, and
keeps host identity out of public output unless explicitly asked. Proven offline end-to-end.
