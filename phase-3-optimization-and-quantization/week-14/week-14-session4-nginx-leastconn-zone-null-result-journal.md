# Week 14 — Session 4: nginx `least_conn` shared-zone fix (null-result)

- **Date:** 2026-06-30
- **Repo touched:** IRS (`inference-reference-stack`) — commit `09bc82b`
- **Scope (declared at open):** apply the `zone` directive to the IRS nginx `workers` upstream; bring up the worker pool; concurrent load-test to verify distribution. **OUT:** `start-stack.sh` review and its T-vs-IRS placement; new serving experiments; speculative decoding.

---

## Summary

Added `zone workers 64k;` to the `workers` upstream block in `nginx/nginx.conf`. The fix is correct by nginx's documented requirements, but the distribution skew it was hypothesized to fix **did not reproduce** on this topology — load was already even across the two 12B workers with *and* without the zone, across serial/concurrent and fresh/keepalive connections. The honest finding is a corrected mental model plus a documentation-correct fix, not a measured before/after improvement. The session's prediction was wrong; this entry records why.

---

## What was done

### 1. Applied the fix
Confirmed beforehand that the `workers` upstream used `least_conn` with **no** `zone` line; upstream name verified as literally `workers` (not assumed). Added `zone workers 64k;`.

### 2. Defeated the inode gotcha (confirmed live, not just feared)
`nginx-frontdoor` is a standalone `docker run --network host` container with a **file** bind-mount of `nginx.conf`. The editor writes via atomic rename, which swapped the host inode (`65407869` → `65408304`). The running container kept serving the **old** inode — `grep` inside it showed the `zone` line absent after the edit. Resolution: `docker restart nginx-frontdoor` re-resolves the bind-mount to the current inode.

- Verified every config change with `nginx -T` (running-config dump), **not** reload-success. Reload success is not evidence the new bytes are live.
- Syntax pre-validated in a throwaway container before each bounce.

### 3. Brought up the worker pool (scope-tight)
Only the two 12B workers — **not** the 31B orchestrator, which a worker-pool distribution test doesn't need. Launched via `start-vllm.sh` (T repo). `google/gemma-4-12B-it-qat-w4a16-ct` was cache-resident; both healthy in ~30s.

| role | GPU | port | container |
|------|-----|------|-----------|
| worker1 | 1 | 8001 | `vllm-worker1-12b-gpu1` |
| worker2 | 3 | 8003 | `vllm-worker2-12b-gpu3` |

**Port drift caught:** the `worker2` role preset hardcodes `:8002`, but the nginx pool expects `:8003`. Launched worker2 with explicit `--port 8003`. This is real T↔IRS drift between the role presets and the pool config — logged as a carry-forward, not a one-off.

### 4. Load-tested the pool
Flooded `:8080` (`/v1/chat/completions`) and counted `$upstream_addr` from `docker logs --since`. Used a throwaway flooder rather than `interference_probe.py` — that tool requires the 31B victim, a committed solo baseline, and a throughput sweep, all overkill for a pool-distribution check.

---

## Key finding — the hypothesized skew did NOT reproduce

Request counts, worker1/worker2 (`:8001`/`:8003`):

| Load shape | NO-ZONE | WITH-ZONE |
|---|---|---|
| fresh-conn c=1 (×3 trials) | 31/29, 31/31, 32/32 | 20/20 |
| fresh-conn c=2 / c=4 / c=20 | 21/21, 41/45, 67/73 | 20/21, 42/42, 60/60 |
| keepalive c=2 / c=4 / c=8 | 18/20, 44/42, 83/81 | 20/20, 41/41, 94/94 |

Distribution is even with and without the zone. With-zone is marginally tighter (exact even); no-zone is within a few percent (noise). No load shape tested produced a skew attributable to the missing zone.

### Two measurement artifacts that nearly produced a fake skew
- The single apparent skew (26/14 at c=1) was a **cold-start transient** — it vanished on repetition (31/31, 32/32).
- The first capture was contaminated by `docker logs` **persisting across `docker restart`** (weeks of history, including old `:8000` orchestrator traffic). Fixed by scoping every capture with `--since`.

### Mechanism — why even private per-worker state still splits evenly
`least_conn` falls back to **weighted round-robin when active-connection counts are tied**. For short requests that complete before the next pick, the counts *are* tied (0/0) at selection time. The round-robin pointer advances per-request within each worker process, so even unshared, per-worker state alternates evenly. The "private counters → pins to the first server" story does not hold for this topology and load.

### Corrected mental model (the session's prediction was wrong on two counts)
The pickup premise expected:
1. serial traffic to **pin to the first server** even with the zone present, and
2. **concurrency to be required** to reveal balancing.

Empirically the opposite held: serial-with-zone was a clean 20/20, and concurrency revealed nothing the tie-fallback didn't already produce. The zone's effect is **unobservable on this topology**, not merely absent — see below.

### Why the zone is unobservable here, and when it would actually bite
The zone defends a specific condition: sustained **asymmetric** active-connection counts where each nginx worker process's private view diverges from global truth. This pool can't manufacture that condition:
- tie-dominated traffic round-robins evenly regardless of shared state, and
- two identical vLLM workers stay symmetric, so the per-worker-view-vs-global divergence never accumulates.

To expose the zone empirically you'd have to **induce backend asymmetry** — slow one worker, or load one directly outside the pool — so active counts stay genuinely lopsided and the no-zone partial views diverge from global. That is a larger experiment with its own go/no-go and does not belong in a close-out. Noted as a future probe, not run.

---

## Decisions

- **Kept the fix.** It is nginx's documented requirement for sharing stateful-balancer state under `worker_processes auto` (12 workers here). Zero-cost; defensively correct for weighted/heterogeneous servers, larger pools, and sustained asymmetric load — even though this symmetric 2-server pool shows no measurable difference.
- **Rewrote the inline comment to match the data.** Dropped the unverified "pins to the first server" claim my own measurements refuted. It now states the documented mechanism plus the empirical even-split result and points to this journal.
- **Did NOT apply the durable directory-mount fix.** `nginx-frontdoor` is a manual `docker run` not captured in any committed script, so converting the mount intersects the OUT-of-scope start-stack launch work. Explicit carry-forward.

---

## Repo state at close

- **IRS:** committed clean (`09bc82b`). Local `main` ahead of origin by 2 (this + the earlier CLAUDE.md commit) — push/hold is an open decision (see carry-forwards).
- **R / T:** clean, up to date (pre-session `git pull --ff-only` both no-ops).
- **Box:** workers torn down at session end; `nginx-frontdoor` left serving the final zone-present config (verified via `nginx -T`). 31B orchestrator (`:8000`) was never up.

---

## Carry-forwards

1. **Directory-mount fix for `nginx-frontdoor`** — convert the file bind-mount to a directory-mount so inode swaps on `git pull` are picked up without a restart. Bundled with the start-stack review (owns the manual `docker run`).
2. **worker2 port drift (T↔IRS)** — the `worker2` role preset hardcodes `:8002` while the IRS pool expects `:8003`. Reconcile in the start-stack / tool-consistency review rather than re-working-around each bringup.
3. **Asymmetric-backend zone probe (optional, future)** — the only setup that would expose the zone's effect empirically: induce sustained backend asymmetry so no-zone per-worker views diverge from global. If run, the throwaway flooder graduates to a proper model-agnostic pool-flooder artifact in T.
4. **IRS push/hold** — local `main` is 2 ahead of origin and unpushed. Decide deliberately so next session's pre-session gate doesn't trip over it.
