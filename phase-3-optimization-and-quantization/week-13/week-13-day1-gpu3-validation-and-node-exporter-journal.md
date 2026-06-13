# Week 13 — Day 1: GPU 3 validation + node-exporter for host metrics

**Date:** 2026-06-13
**Phase:** 3 — Optimization & Quantization
**Host:** `inference` (4× RTX 3090; GPUs 0+2 NVLink-bridged, GPUs 1+3 on PCIe 3.0 x1)
**Repos touched:** `rtx3090-ai-training` (no commits today), `inference-reference-stack` (1 commit: `047c9be`)

---

## What this day was and wasn't

The Day-1 plan (from the Week 13 pickup) had three parts: stand up a second 12B
sub-agent worker on GPU 3, run the **boot-choreography experiment** (the day's headline
measurement), and put both 12B workers behind a basic nginx front door.

In reality the day was mostly **preparation and validation** for the boot experiment,
plus two useful findings that fell out along the way. Specifically:

- The boot choreography and nginx front door did **not** start.
- The GPU-3 work, on reflection, was **not** a "second worker" test. No worker was
  running when we brought GPU 3 up (pre-session checks showed all four GPUs idle), so
  nothing concurrent was exercised. What we validated is narrower: that the 12B QAT
  model runs on GPU 3 — the *other* PCIe-x1 card — the same way it ran on GPU 1 in
  Week 12. That was an expected result on identical hardware, logged for completeness.
- The genuinely reusable outcomes of the day were: characterizing the launcher's
  defaults, re-confirming a Week 12 endpoint constraint the hard way, and standing up
  **node-exporter** so the boot experiment can actually be measured.

Concurrent co-residency of multiple workers (let alone workers + the 31B orchestrator)
remains **untested** — that is precisely what the boot-choreography experiment exists to
exercise, starting from a clean-slate boot next session.

Background, for a future reader: the architecture under construction is a two-tier
agentic system. A larger "orchestrator" model (Gemma 4 31B, served across the two
NVLinked GPUs) splits work into sub-tasks and hands them to smaller, faster "worker"
models (Gemma 4 12B QAT, one per PCIe-x1 card). Week 12 validated a single worker on
GPU 1. Today validated GPU 3 as an equally viable single-worker host.

---

## 1. Pre-session checks — clean

- **Model cache present:** 12B QAT weights resident (9.6 GB on disk), 31B FP8 present.
  No eviction since Week 12.
- **GPUs idle:** no leftover compute processes on any of the four cards — including no
  worker from Week 12 (this is why today's GPU-3 work was single-worker, not concurrent).
  The four `irs-*` containers from the observability stack were all `Exited` (the
  standalone `irs-vllm` had exited (1) three weeks ago — harmless leftover; we launch
  workers directly via `start-12b-qat.sh`, not through that compose service).
- **Image digests verified:** `vllm/vllm-openai:gemma4-unified` matched the pinned
  `sha256:e828735f…63ed450`; `v0.21.0` (the 31B image) present.
- **venv:** the environment is at `~/work/rtx3090-ai-training/ai-inference`, **not**
  `.venv` as the pickup checklist guessed. `httpx 0.28.1` confirmed once activated.
- **git:** both repos clean at the start.

## 2. Aside — vLLM 0.23.0 check (Day-4 convergence input, logged not acted on)

The pickup says to check the vLLM release page a couple of times per week for the
eventual version-convergence work (the goal of getting **one** image that serves both
the 12B `gemma4_unified` model and the 31B FP8 model, so the temporary patch and the
custom preview image can both retire). Checked today.

- **v0.23.0 exists** (PyPI, ~1 week old).
- **The decision-relevant signal:** native encoder-free `gemma4_unified` support
  appears to have landed in mainline vLLM (PR #44429). That is exactly the convergence
  enabler — if it's in a mainline image that also carries the 31B FP8 path, the 3-line
  prefix patch and the custom `gemma4-unified` image could both retire onto one pinned
  mainline image. The exact release that carries it (0.23.0 vs 0.22.0) wasn't pinned
  down today and should be confirmed against the release notes at Day 4.
- **Caveat keeping it an open question, not a done deal:** there are *fresh* open bugs
  on the Gemma-4 **multimodal** path (an audio-profiling dimension-mismatch crash on
  0.22.1; a vision-config warning 4 days ago). We run **text-only** (multimodal inputs
  disabled — that's the weight-footprint lever from Week 12), so these may not affect
  us, but "may not" is precisely what the Day-4 convergence evaluation exists to settle
  empirically, with the Week 11 31B re-baseline as the acceptance test.
- One reassuring detail: a recent issue log shows the native path resolving the
  architecture as `Gemma4UnifiedForConditionalGeneration`, reporting max model len
  262144 and forcing the `TRITON_ATTN` backend due to heterogeneous head dimensions
  (head_dim 256 / global_head_dim 512) — which matches our Week 12 characterization
  exactly. So the native path behaves consistently with what we measured.

**Action taken:** none. Nothing pinned or pulled. The environment is unchanged.

## 3. GPU 3 validation + two findings

Goal: confirm the 12B QAT worker runs on GPU 3 (the second PCIe-x1 card) and matches
the validated GPU-1 configuration. Operational convention: the worker on GPU *N* serves
host port 8000+*N*, so GPU 3 → port 8003.

The headline result is unremarkable, as expected: **the 12B QAT model runs on GPU 3
identically to GPU 1.** GPU 1 and GPU 3 are identical hardware on identical PCIe x1
links, and single-GPU (TP=1) serving has no inter-GPU traffic, so the x1 link — the
thing that makes those cards painful for parallel work — is irrelevant here. There was
no reason to expect a difference and none appeared. Logged for completeness.

Two findings along the way *were* worth the day, though:

### Finding A — launcher defaults (characterization of `start-12b-qat.sh`)

- **max-model-len defaults to 32768.** First launch (`--gpu 3` with no MML flag) came up
  at 32768, not the validated 131072. The launcher only changes MML if `--max-model-len`
  is passed explicitly; it is not derived. ("Max-model-len," MML, is the largest
  prompt+response token count the server accepts. Week 12 validated 131072 — the model's
  architectural position-embedding limit; memory was never the binding constraint on
  this worker.) Correct usage: pass `--max-model-len 131072` explicitly.
- **Host port is not derived from `--gpu`.** The script defaults `PORT` to 8001 and only
  changes it via `--port`. The 8000+N convention lives in operational discipline, not in
  the tool. For a multi-worker layout that's a foot-gun — easy to forget `--port` and
  collide two workers. A `--gpu`-derived default would close it. **Parked** as a
  candidate launcher change (Day-4 launcher-touch list, alongside the pending
  CUDA-graph-tax flag).
- **False alarm, recorded so it isn't re-investigated:** the vLLM startup log says
  `Starting vLLM server on http://0.0.0.0:8000`. That is the **container-internal** port
  — vLLM always listens on 8000 inside the container, and the launcher maps it to the
  host port via `-p "${PORT}:8000"`. `docker ps` correctly showed the host binding as
  `0.0.0.0:8003->8000/tcp`. The internal-8000 line is expected and not a collision.

### Finding B — chat endpoint, *again* (re-confirmed Week 12 constraint)

We already established in Week 12 that this worker must be probed (and called) via
`/v1/chat/completions`, **not** raw `/v1/completions`: the raw path triggers a known
Gemma-4 token-repetition failure mode (degenerate runs of repeated characters) because
it skips the chat template and beginning-of-sequence token handling. Despite that being
a documented Week 12 finding, today's first functional probe used raw completions anyway
— and produced exactly that gibberish (`d711114111_r1111`). We forgot the prior finding
and tripped over the same thing a second time before switching to the chat endpoint,
which returned a clean `"Understood"` with a normal stop. Logging this plainly because
the fact that it bit *twice* is the argument for encoding the chat-endpoint assumption
somewhere the orchestrator can't skip past it.

### Verification of the GPU-3 run

| Check | Result |
|---|---|
| Host port | `0.0.0.0:8003->8000/tcp` |
| Physical GPU placement (uuid-join) | PID 419876 → `GPU-0da3bc5b…` → **GPU 3** ✓ |
| GPU memory | ~22,152 MiB (≈ util 0.90 of 24 GB) |
| Functional probe (`/v1/chat/completions`, nonce, warm) | `"Understood"`, `finish_reason: stop` ✓ |
| max-model-len | 131072 (matches validated GPU-1 config) |

"uuid-join" = matching the process's reported GPU UUID against `nvidia-smi -L` to
confirm physical placement rather than trusting the launcher's intent.

## 4. node-exporter — host-metrics instrumentation for the boot experiment

The boot-choreography experiment (next session) is about **host** resource contention:
when multiple model servers boot together, they compete for the single NVMe disk
(reading tens of GB of model files), the 64 GB of RAM, and CPU. The existing
observability stack only has **dcgm-exporter**, which watches the GPUs. Nothing was
watching the host. So host visibility had to be added first, or the experiment couldn't
be measured. This was the main deliberate work of the day.

**node-exporter** is the standard tool for host CPU/RAM/disk metrics. Added it to the
`inference-reference-stack` compose stack (same pattern the pickup notes flagged for the
upcoming nginx addition — a small infra piece that became load-bearing).

- **Two files edited:** the compose file (new `node-exporter` service) and the
  Prometheus scrape config (new `node` job). Both follow existing conventions:
  `restart: unless-stopped`, localhost-bound exporter port (9100), reached by
  service-name DNS on the `irs` bridge network as `node-exporter:9100`.
- **`pid: host` + host mounts:** node-exporter needs the host PID namespace and
  read-only mounts of `/proc`, `/sys`, and `/` to report the *true host* view rather
  than its own container's namespaced view. This is the conventional, documented setup;
  it's a broader grant than the other services hold, intentional and standard for a
  single-host node-exporter. Without it the boot-contention numbers would be wrong.
- **Image:** `prom/node-exporter:v1.8.2`.

**Workflow:** edits made on the Mac → committed → pushed → pulled to `inference`. Brought
up with `docker compose up -d node-exporter` (named service only — a bare `up -d` would
have also started `irs-vllm` on GPUs 0+2, the Week 8–9 reference model we don't want
running).

**Verification:**
- Host metrics scrape correctly: real NVMe mounts visible (`/dev/nvme0n1p2` at `/` with
  ~1.24 TB free), `node_memory_MemAvailable_bytes` ≈ 55.7 GB. At the time of this check
  the GPU-3 worker was the only model server running, on a 64 GB box — so ~55.7 GB
  available implies a single 12B worker's host footprint is roughly 8 GB of RAM, and the
  exporter *sees* it, confirming the host-view wiring works. (This is a one-worker
  figure; multi-service host footprint is still to be measured in the boot experiment.)
- Prometheus scrapes the `node` job over the `irs` network: `"health": "up"`, no errors.
- Committed as `047c9be` on `inference-reference-stack`; `HEAD`, `origin/main`,
  `origin/HEAD` aligned, working tree clean. Commit-before-running boundary satisfied —
  any boot-experiment results JSON from here will record a clean SHA.

Grafana was not started (not needed for collection; only needed to *view* the data
later). Prometheus + node-exporter are the live collection path.

---

## State at end of Day 1

- **12B worker:** GPU 3 validated as a viable single-worker host, identical behavior to
  GPU 1 (Week 12). MML 131072, util 0.90, chat endpoint. **Currently still running**; it
  will be torn down at the start of the boot experiment, which needs a clean-slate boot.
- **No concurrency tested:** at no point today were two workers (or workers + the 31B)
  resident at once. Co-residency is entirely ahead of us.
- **Observability:** node-exporter live, Prometheus scraping host + GPU metrics.
  Committed and pushed (`047c9be`).
- **31B orchestrator:** not brought up this week yet. Joins in the boot experiment / Day 2.
- **No work started on:** boot choreography, nginx front door.

## Pending / next session

1. **Boot choreography (Day-1 headline, deferred).** Staggered vs simultaneous full-stack
   boot (two 12B workers + 31B TP=2). This is also the **first concurrency test** of the
   layout. Capture per-instance time-to-healthy and host metrics across the boot window
   (now possible via node-exporter): NVMe throughput/util, RAM + page cache +
   per-process RSS + swap, CPU. Produce a documented boot procedure. **Also capture the
   steady-state host RSS of all three resident services** — three vLLM processes on 64 GB
   is itself an unverified fit, and that number bounds the layout's host headroom.
   Starts from a clean teardown of everything, including the GPU-3 worker.
2. **nginx front door (basic).** Two 12B workers in a `least_conn` upstream pool plus
   direct named routes per worker. Functional probes through pool and named routes.
3. **APC (automatic prefix caching) verification probe.** Same long prefix (≥32K) sent
   twice to the *same* worker; compare time-to-first-token. Methodology inversion: this
   probe *wants* cache hits, so **no nonce** (do not reuse the sweep defaults that
   deliberately defeat caching). A null result is a real finding — it would collapse the
   affinity design to `least_conn`-only.

## Parked items (not blocking)

- **Launcher: `--gpu`-derived default port** for `start-12b-qat.sh`, to remove the
  forget-`--port`-and-collide foot-gun. Day-4 launcher-touch candidate.
- **CUDA-graph-tax flag** (`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`) for
  `start-vllm.sh` — still pending from Week 11, flag-gated opt-in. Day-4 decision.
- **vLLM version convergence** — 0.23.0 looks more promising (native `gemma4_unified`
  appears in mainline) but stays a Day-4 empirical evaluation gated on the 31B
  re-baseline; text-only-vs-multimodal bug exposure is the specific thing to probe.

## Notes for the orchestrator design (banked from today)

- **The worker tier must be called via `/v1/chat/completions`, never raw
  `/v1/completions`.** This is a hard constraint (Gemma-4 PT-path token repetition),
  re-confirmed today the hard way. Worth encoding somewhere the orchestrator can't skip.
