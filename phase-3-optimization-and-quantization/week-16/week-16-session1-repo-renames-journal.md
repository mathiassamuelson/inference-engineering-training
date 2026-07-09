# Week 16 — Session 1: repo renames

**Date:** 2026-07-09
**Session type:** repo maintenance (Claude Code, inference host). Renames + cross-reference
sweep + verification. No measurement, no serving.
**Deliverable:** both repos renamed, remotes rewired on the inference host, cross-reference
sweep committed and pushed, redirects verified. This is the **first** Week 16 session — the
renames come before any final document is drafted so every closing document carries the final
names.

---

## Scope (declared at open)

- **IN:** rename both repos on GitHub; rewire remotes on the inference host; sweep all three
  repos (R, T, IRS) for old-name cross-references and update the live ones; rename the local
  checkout directories; verify redirects and fetch/push; commit and journal. Bundled the carried
  `start-stack.sh` 31B-default fix (Mathias's call at open).
- **OUT:** journal consolidation, the capstone summary, the method Pulse (later Week 16
  sessions); the successor program and anything Blackwell / vLLM-0.24; any measurement or
  serving work; rewriting any historical journal.

## The renames

| | old | new |
|---|---|---|
| R | `rtx3090-ai-training` | `inference-engineering-training` |
| T | `rtx3090-ai-training-tools` | `ai-training-tools` |

**Rationale:** the program's identity anchors to the role and the work — inference engineering —
not the RTX 3090 hardware that happened to run it. GitHub creates permanent redirects on rename
(web + git), so published Pulses and the upstream vLLM issue reference keep resolving. The old
names are never reused (reuse would break the redirect).

## Pre-session state

Gates clean. `git pull --rebase` on R and T (R pulled in the Week-15 close-out —
`docs/delegation-architecture.md` + the day-3 journal — confirming it was on origin),
`git pull --ff-only` on IRS (already up to date). Trees clean on all three.

## Decisions settled at open (Mathias)

1. **GitHub rename mechanism:** Mathias renamed both repos in the GitHub web UI (Settings →
   rename). This box has no `gh` CLI and no HTTPS token (remotes are SSH-only), so the
   irreversible outward-facing step was his to own regardless.
2. **Local checkout directories:** renamed too (`~/work/rtx3090-ai-training` →
   `~/work/inference-engineering-training`, and the tools dir likewise) for full name
   consistency — not just remote rewiring. This carried a venv-repair cost (see below).
3. **Bundled `start-stack.sh` fix:** yes, folded into this session's T commit.

## Cross-reference sweep — dispositions

Swept all three repos with `git grep -E 'rtx3090-ai-training'`.

**R (29 hits).** Updated the live docs only:
- `CLAUDE.md` — the two repo-name bullets + the run-convention `cd`/`T=` example.
- `README.md` — the two-repositories list (names + GitHub URLs), the run-convention paths, and
  the Quick-start clone URLs + `cd`.
- `docs/training-plan.md` — the Week-14 repo-split entry's `T` name (live curriculum → current
  truth).
- **Left untouched (never-rewrite):** the Week-13/14 historical journals, the committed result
  JSONs (absolute `victim_sweep_file` / `baseline_file` / input-path fields), and the three
  frozen **phase-1 scripts** (`diagnose_cuda_mismatch.sh`, the two `week-02-tensorrt/*.sh`).
  The phase-1 scripts embed old-name absolute paths — one is already `~/rtx3090-ai-training`
  (no `work/`), i.e. pre-existing frozen state. The local-dir rename makes those paths stale,
  but they are historical artifacts; a phase-1 script would only need a one-line path edit if it
  is ever re-run.

**T (9 hits).** Updated `README.md` (title + R URL + run paths), `requirements.txt` header,
`tools/provenance.py` docstring, `tools/throughput_sweep.py` comment. No hits left.

**IRS (0 hits).** The anticipated IRS references don't exist — nothing to change. Also confirmed
the LinkedIn README's source links are **relative** (`./file.md`), so no repo name is embedded
there either. Both README program-title lines still say "RTX 3090" — those name the *hardware*,
not the repos, and are intentionally unchanged (out of the rename sweep's scope).

## Bundled fix — `start-stack.sh` 31B defaults

The launcher's `start-stack.sh` still carried stale 31B defaults from before the QAT migration:
- `MODEL_31B` `RedHatAI/gemma-4-31B-it-FP8-block` → `google/gemma-4-31B-it-qat-w4a16-ct`
- `MML_31B` `33024` → `131072`
- `UTIL_31B` stays `0.95` (already correct); refreshed the now-stale 33024/FP8 comments and the
  usage example. These now match the `start-vllm.sh` orchestrator role preset (the source of
  truth for production placement).

## Local-directory rename + venv repair

`mv`'d both checkouts. The R venv (`ai-inference/`, gitignored, lives inside R and moved with it)
embeds absolute paths; repaired **20 files** (19 under `bin/` — `activate*` scripts and console
shebangs — plus `pyvenv.cfg`'s `command` line) by rewriting
`work/rtx3090-ai-training/ai-inference` → `work/inference-engineering-training/ai-inference`.
Verified: `source ai-inference/bin/activate` sets `VIRTUAL_ENV` and `sys.executable` to the new
path; `python --version` = 3.12.3. No profile/rc references the old path, so nothing else needed
updating (re-activate any interactive shells that had the old venv on `$PATH`).

## Verification

- **New remotes (inference host):** `git remote set-url origin` → new URLs on both. `ls-remote`
  on each new URL resolves directly to the just-pushed HEADs (R `ceecb14`, T `9801227`).
- **Old git URL redirect:** `git ls-remote git@github.com:.../rtx3090-ai-training.git` (and the
  tools URL) still return the new HEAD SHAs via GitHub's rename redirect — demonstrated live when
  the initial push to the old URL printed the new repo name and landed.
- **Old web URL redirect:** both old `https://github.com/...` URLs return **HTTP/2 301** →
  the new names.
- **Trees clean** on R and T from the renamed dirs; fetch dry-run rc=0 on both.
- `gh repo view` was not run (`gh` not installed on this box); the ls-remote + 301 checks cover
  the same ground (repo reachable under the new name, old name redirects).

## Commits

- **T** `9801227` — rename sweep + `start-stack.sh` 31B-default fix.
- **R** `ceecb14` — rename sweep in live docs.
- Both pushed (to the old URLs, which redirected onto the renamed repos; then remotes rewired to
  the new URLs). This journal committed on top.

## Carried to later Week 16 sessions / follow-ups

- **Mac mini remote rewire — NOT done here.** This session ran on the inference host; I can't
  reach the Mac from here. On the Mac, run
  `git remote set-url origin git@github.com:mathiassamuelson/inference-engineering-training.git`
  (and `.../ai-training-tools.git` for T). Day-to-day remotes should point at the real names, not
  lean on the redirect. Its local checkout dirs are the Mac's to rename if desired.
- **Phase-1 scripts** retain old-name absolute paths by design (frozen historical artifacts);
  edit only if re-run.
- Remaining Week 16 sessions (unchanged sequence): journal consolidation → capstone summary →
  method Pulse → plan closure.

## Close-out

R / T / IRS clean at close. Both repos renamed; inference-host remotes rewired; cross-reference
sweep committed and pushed; redirects and fetch/push verified.
