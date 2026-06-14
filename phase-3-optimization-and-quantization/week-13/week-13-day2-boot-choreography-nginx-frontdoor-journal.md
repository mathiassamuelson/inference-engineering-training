# Week 13 — Day 2: Boot Choreography + nginx Front Door

**Date:** 2026-06-13 (work ran into the early hours of 06-14 UTC)
**Phase:** 3 — Optimization & Quantization
**Focus:** First concurrent boot of the full multi-tier stack, and a minimal nginx front door over the worker tier.

---

## Summary

Two experiments closed today. First, **boot choreography**: bringing up the whole serving
stack — two small (12B) workers plus one large (31B) model split across two GPUs — and
measuring whether booting them one-at-a-time ("staggered") or all-at-once ("simultaneous")
is faster, and whether three model servers even fit in 64 GB of host RAM at the same time.
This was also the **first time the full layout ran concurrently** at all, so it doubled as a
go/no-go on the whole co-residency design. Verdict: **it works** — all three boot together,
host RAM has ~40 GB to spare, no swapping, and simultaneous boot is ~2.5× faster on
wall-clock.

Second, a **minimal nginx front door**: a single entry point that load-balances across the
two workers and also offers direct "give me this specific instance" routes. The routing
contract verified cleanly after a couple of instructive failures.

Both experiments produced prediction misses that are worth more than the passes — they
corrected wrong mental models (see the "Predictions vs reality" sections).

---

## Starting state

- All four GPUs idle at session start (the previous day's worker had been torn down between
  sessions). Clean slate confirmed before any launch.
- Observability stack (Prometheus + node-exporter, added Day 1) already running and
  collecting host metrics.
- `irs-vllm` (the Week 8–9 reference model on the NVLink GPU pair) kept **down** — it would
  compete with the 31B for the same two GPUs.

**Glossary for this entry** (defined once, used throughout):
- **TP=2 (tensor parallelism, degree 2):** one model split across two GPUs that share the
  work on every layer. Our 31B model runs TP=2 on the NVLink-bridged pair (GPUs 0 and 2).
- **MML (max model length):** the largest context window (prompt + output tokens) a server
  is configured to handle.
- **KV cache:** GPU memory that holds the running attention state for in-flight requests.
  It is *residual* — it gets whatever GPU memory is left after weights, CUDA graphs, and
  activations are accounted for.
- **util (`--gpu-mem-util`):** the fraction of a GPU's memory vLLM is allowed to claim.
  Higher util → more KV cache → larger serviceable context.
- **CUDA-graph memory-profiling tax:** a behavior introduced in vLLM 0.21.0 that reserves
  extra memory up front, which *reduces* the effective util available to KV cache. At
  util 0.90 it makes the engine behave as if util were ~0.86.
- **compile / capture (the two startup phases):** after weights load but before the server
  answers requests, vLLM runs `torch.compile` (traces the model's forward pass and generates
  fused, optimized GPU kernels — a one-time cost that makes every later token faster) and
  then **CUDA-graph capture** (records sequences of GPU operations so they can replay as a
  single unit, removing the per-operation CPU launch overhead during token generation). Both
  do more work at larger context sizes, so together they dominate boot time. The capture
  phase's up-front memory reservation *is* the CUDA-graph tax above — so it costs both boot
  time and KV memory.
- **time-to-healthy:** seconds from launching a server to it answering a real request.

---

## Experiment 1 — Boot choreography

### What and why

The three tiers live on **disjoint GPUs**: the 31B on GPUs 0+2 (the fast NVLink pair),
worker-1 on GPU 1, worker-2 on GPU 3. Because no two tiers share a GPU, there is no
GPU-memory contention between them — so this experiment is really about **host-side**
contention: a single NVMe disk streaming all the model files, 64 GB of system RAM holding
three server processes plus disk cache, and CPU contention during the compute-heavy
"compile and capture" phase each server runs at startup.

The question: launch the three servers **staggered** (one after the previous is healthy) or
**simultaneous** (all at once)? Simultaneous should overlap the slow phases and finish
sooner — *unless* the overlapping memory demand pushes the host into swap, which would
invert the result into thrashing.

### The tool

Built `tools/start-stack.sh`, a single orchestrator that takes a mode argument
(`staggered` | `simultaneous` | `teardown`), launches all three tiers by calling the
existing per-tier launch scripts, and then:
- probes each server's **chat endpoint** to time time-to-healthy (the chat endpoint is
  mandatory for these models — the raw completions endpoint produces gibberish);
- verifies GPU placement empirically by cross-referencing `nvidia-smi` process and GPU
  lists (never trusting the launcher's intent alone);
- records steady-state memory and writes a self-describing JSON whose filename and contents
  capture which models were tested.

### Bugs found and fixed (the script took four passes)

This is logged in full because each was a real, generalizable trap:

1. **Launcher interface mismatch.** The orchestrator initially guessed the per-tier
   launchers' flags. Reconciled against their actual argument parsing: the memory-util flag
   is `--gpu-mem-util` (not the vLLM-native long name), tensor parallelism is set via
   `--mode tp --size 2`, and — critically — both worker launchers default to the *same*
   container name, so launching two without distinct `--name` values collides and the
   second fails. Fixed by passing explicit per-instance names.

2. **The large-model launcher hard-coded an interactive terminal flag (`-it`).** That works
   when a human runs it in a terminal but fails the instant anything launches it
   non-interactively ("the input device is not a TTY"). Fixed by making the terminal flag
   conditional on whether a real terminal is attached — so interactive use is unchanged, but
   scripts/CI work too.

3. **The orchestrator's timeout check never fired.** A subtle shell bug: the elapsed-time
   comparison was written in a way that the tool interpreted as *writing to a file named
   after the timeout value* instead of *comparing against it*. Result: instead of failing
   fast at the timeout, the script polled forever and looked hung. Fixed to a proper
   comparison. (This is why an earlier attempt appeared to hang for 18 minutes — it would
   have polled indefinitely.)

4. **The 31B refused to boot at util 0.90.** Not a script bug — a real config finding. At
   util 0.90, the vLLM 0.21.0 CUDA-graph tax cut the effective util to ~0.86, leaving only
   2.86 GiB of KV cache, which tops out at a ~23,600-token context — *below* our 33,024
   baseline. The engine correctly refused. Fixed by raising the 31B to **util 0.95** (the
   value Week 11 characterized the 33,024 baseline at), which leaves ~4 GiB KV cache, well
   clear of the requirement. **Important:** this is independent of co-residency — the
   31B's GPUs are disjoint from the workers', so it would have failed solo too.

5. **A field-count mismatch crashed the post-launch bookkeeping.** When the internal service
   table was widened to carry container names, one parsing loop still read the old number of
   fields, so a port value got contaminated with a name and a `docker` lookup choked. The
   boot itself was fine and all three servers were healthy; only the results-writing step
   died. Fixed the parse; the next run completed end-to-end.

### Results

Timings are seconds to time-to-healthy. "Wall-clock" is total elapsed for the whole stack.

| Mode         | worker-1 | worker-2 | 31B (TP=2) | Stack wall-clock |
|--------------|---------:|---------:|-----------:|-----------------:|
| Staggered    |   145.4  |   137.4  |     107.4  |          ~390 s  |
| Simultaneous |   154.0  |   149.1  |     107.8  |           154 s  |

Host memory (from `free`, the authoritative host-level measure):

| Mode         | Available RAM (steady) | Swap used |
|--------------|-----------------------:|----------:|
| Staggered    |               ~40.1 GB |     48 MB |
| Simultaneous |               ~40.1 GB |     52 MB |

Placement verified on both runs: 31B on GPUs 0+2 (identical ~23.5 GB each — symmetric TP=2
ranks), workers on GPUs 1 and 3 (~22.2 GB each). Nothing strayed.

### What the numbers mean (findings)

- **Simultaneous wins wall-clock (~2.5×), but not for free.** Each *individual* worker's
  boot time rose ~6–8% under simultaneous (145→154, 137→149) because all three contend for
  the single NVMe during the overlapping file-load phase. The 31B was untouched (107→108):
  it launched into a cold cache and won its disk read before contention built, and its
  later compile/capture phase is compute-bound, not disk-bound. So the bottleneck is the
  **single disk during loading**, exactly as hypothesized — not RAM, not CPU.

- **Host RAM is not the binding constraint.** Three servers co-resident leave ~40 GB
  available and never meaningfully touch swap. The 64 GB host comfortably fits the layout.
  (Note: per-container memory as reported by `docker stats` is unreliable here — it includes
  reclaimable disk cache and swings with read history. The honest host-pressure number is
  `free`'s used/available, which was stable at ~22 GB used / ~40 GB free across both modes.)

- **Boot time is dominated by compile/capture** — the `torch.compile` kernel-generation and
  CUDA-graph-recording phases every server runs at startup (see glossary) — **and these scale
  with context size.** The 31B booted *faster* than the workers (107 vs ~140 s) despite being
  the larger model — because the workers run at a 131,072-token context while the 31B runs at
  33,024, and the startup capture cost grows with context length.

- **Page-cache warmth is visible and real.** In the staggered run, worker-2 booted ~8 s
  faster than worker-1 (137 vs 145) and read **0 bytes** from disk — its model files were
  already cached in RAM from worker-1's read moments earlier. Two independent signals (boot
  time and disk bytes), one cause.

- **First confirmation the disjoint-GPU design has no VRAM cross-contention.** The 31B's KV
  budget on GPUs 0+2 is identical whether or not the workers are running on GPUs 1+3 — the
  util-0.90 boot failure proved this by being independent of the workers entirely.

### Predictions vs reality (recorded honestly)

- Predicted workers ~70–120 s; actual ~140–145 s (**~20–25% over**). Cause: underweighted
  the compile/capture cost at the 131K context. Reproducible across runs, so it's a stable
  miss, not noise.
- Predicted 31B ~90–150 s; actual ~107 s (**in range**).
- Predicted three-server footprint ~26–30 GB; actual ~22 GB used (**overestimated** — the
  workers are leaner than modeled).
- Predicted "simultaneous wins iff peak RAM avoids swap." **Confirmed on the favorable
  branch** — peak load stayed clear of swap because steady-state already had ~40 GB headroom.

---

## Experiment 2 — nginx front door

### What and why

A single HTTP entry point (port 8080) in front of the two workers, doing two jobs:
1. a **load-balanced pool** so a generic client can hit "any worker" and be distributed
   across both (using nginx's `least_conn` policy — route each request to whichever backend
   has the fewest active connections);
2. **named instance routes** so a client can deliberately target one specific server.

The verification instrument is the nginx access log: it records which backend served each
request, so we can *see* routing rather than guess.

### The routing contract

Settled on **version-first** paths (the version prefix at the root, matching common API
convention and giving a clean place to reject unsupported versions):

| Path                       | Goes to                              |
|----------------------------|--------------------------------------|
| `/v1/chat/completions`     | load-balanced pool over both workers (any `/v1/` endpoint that isn't an instance route) |
| `/v1/worker/1/...`         | worker instance 1 (port 8001)        |
| `/v1/worker/2/...`         | worker instance 2 (port 8003)        |
| `/v1/orchestrator/1/...`   | the 31B (port 8000)                  |
| `/healthz`                 | liveness check                       |

The number is an instance index within a tier (so a third worker would be `/v1/worker/3/`),
and hardware (which GPU) is deliberately kept out of the path. The named routes strip their
prefix and re-add `/v1/` for the backend, since the model servers themselves serve under
`/v1/`.

### Findings

- **Named routes pin deterministically** — each targets its own backend, every time.
- **The pool only load-balances under concurrent traffic.** `least_conn` routes by *active*
  connection count; when requests are sent one-at-a-time and each finishes before the next
  arrives, every backend shows zero active connections, and on that tie nginx picks the
  **first** server in the list — so six serial requests all landed on worker-1. Sent six
  *concurrently* instead, they spread across both workers. This is a property of
  `least_conn`, not a misconfiguration: **it does not round-robin on ties.** Real
  distribution characterization needs genuine concurrent load (Day 3).
- **A negative control confirmed the version namespace works:** an unsupported `/v2/...`
  request is rejected at the door (nginx 404, never proxied to a backend).
- **Stale-mount trap (the day's sharpest ops lesson).** The nginx config is bind-mounted as
  a *single file*. Git updates files by writing a new file and renaming it over the old one,
  which changes the file's underlying identity (inode). A single-file bind mount stays
  pinned to the *original* inode, so after a `git pull` the running container still sees the
  **old** config — and `nginx -s reload` faithfully reloads that stale content with no
  error. The fix is `docker restart` (which re-resolves the mount), not reload. This bit us
  live: the first version-first probe returned all-404s because the container was still
  routing on the previous contract. Confirmed twice (the running config dump showed the old
  routes; the probe showed old behavior), then fixed by restart.

### Predictions vs reality (recorded honestly)

- Predicted the serial pool probe would alternate across workers. **Wrong** — `least_conn`
  picks the first server on a tie; it is not round-robin. Corrected by a concurrent re-test.
- Predicted `nginx -s reload` would pick up the new config. **Wrong** — single-file bind
  mounts need a container restart. Corrected by `docker restart`.

Both misses are logged as findings, not footnotes — each corrected a wrong mental model.

---

## What got committed

- `tools/start-stack.sh` — the boot orchestrator (across its fixes), committed before the
  measured runs so the recorded git SHA in each result file is clean.
- Boot-choreography result JSONs (staggered + simultaneous) and a results-dir `.gitignore`.
  The raw `iostat`/`free` capture logs are **intentionally not versioned** — the repo-root
  `.gitignore` excludes `*.log`, and the JSONs already carry the summary numbers; the logs
  were backup evidence for the disk-saturation and swap-flat claims.
- `nginx/nginx.conf` — committed twice: the initial pool+named-routes config, then the
  version-first revision. This repurposed the Week 8–9 `irs-vllm` front-door config (which
  is fine, since that stack stays down this week).

**Provenance note:** the simultaneous-mode JSON's recorded git SHA was captured while the
staggered result files were still uncommitted, so that SHA reflects a results-dir that
wasn't fully clean. The *code* (`start-stack.sh`) was committed and unchanged across both
runs, so the measurement is valid; this is bookkeeping dirtiness only.

---

## Deferred / parked

- **APC (automatic prefix caching) verification probe** — the third Day-2 "if time allows"
  item; not reached. Moves to Day 3.
- **nginx directory-mount robustness fix** — mounting the config's *directory* instead of
  the single file would survive git's inode swap and let `nginx -s reload` work. A small
  Day-4-style tool touch; for now the rule is "restart, don't reload."
- **`start-stack.sh` dirty-tree warning is a false positive for the results directory** —
  it warns when the working tree is dirty, but result files are *expected* to be uncommitted
  at write time. Worth teaching it to ignore the results dir. Day-4 tool touch.
- Existing Week-11/Day-4 carry-overs remain parked (version convergence, the CUDA-graph-tax
  opt-in flag decision, the worker launcher's port-derivation fix).
