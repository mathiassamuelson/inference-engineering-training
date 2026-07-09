# Week 10 Summary — inference-reference-stack born: Compose scaffold + observability layer

**Dates:** Day 1 2026-04-13; Day 2 undated in its journal (committed 2026-04-25). No further Week 10 sessions were journaled.
**Model served:** `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`, TP=2 on GPUs 0+2, MML 16,384
**Engine:** `vllm/vllm-openai` pinned by full digest (`sha256:0cb12dc9…` — the exact Week 9 image, preserved deliberately)

## TL;DR

Week 10 pivoted from runtime optimization to production-deployment scaffolding and produced the third repository of the program: **`inference-reference-stack`** (public, Apache 2.0), a Docker Compose stack with vLLM, Prometheus, Grafana, and DCGM-exporter running by Day 2. The planned Triton serving layer was dropped on Day 1 for a concrete version conflict, documented in-repo. The week is infrastructure, not measurement — no throughput or load characterization was performed, and the nginx/TLS front-door items declared in Compose were deferred and never launched within Week 10 (the front door next appears in the record at Week 13 Day 2).

## Day 1 — Compose scaffold and the Triton pivot

The new repo landed with a four-service `docker-compose.yml` (vllm, nginx, prometheus, grafana), a streaming-aware nginx config (buffering disabled on the completions endpoints — the Week 9 footgun pre-empted), Prometheus scrape config, Grafana provisioning, and an `.env` template. Only the `vllm` service was started on Day 1; it loaded the 26B MoE clean and answered `/v1/models`, `/v1/chat/completions`, and `/metrics`.

**The Triton pivot (decision, documented in the repo's "Why not Triton?" README section):** the current NGC Triton-vLLM image bundled vLLM 0.15.1, three minor releases behind the 0.18.2rc1 build the Gemma 4 AWQ-INT4 path required. Options were to gamble on 0.15, build a custom layer, or drop Triton for standalone vLLM's OpenAI server. Standalone vLLM won on two grounds: the repo's architectural value (proxy, observability, gateway) lives *above* the engine and doesn't care which engine it is, and keeping the exact Week 9 image preserved the pending #39133 before/after re-test as an uncontaminated asset. Triton was left explicitly on the table for a future multi-model/ensemble need.

Two durable operational choices from Day 1:

- **Digest pinning as the reproducibility anchor** — when the #39133 fix landed, the digest could be swapped in isolation, attributing any change to the fix rather than version drift. (This is the practice that later became the pinned-image convention of the production stack.)
- **Compose bridge network, deliberately not `--network host`** — so nginx and Prometheus resolve `vllm:8000` via service DNS. Port 8000 bound loopback-only on the host from the start, with nginx intended as sole external ingress.

Minor detour, kept for reference: Ubuntu's `docker.io` package needs `docker-compose-v2` (not `docker-compose-plugin`, which is docker-ce's name for it) to get Compose V2.

## Day 2 — Prometheus, Grafana, DCGM

End state: four services up on the `irs` bridge network (vllm + prometheus v2.54.1 + grafana 11.3.0 + dcgm-exporter), nginx still declared-not-launched. The starter dashboard (`observability/grafana/dashboards/vllm-overview.json`) is auto-provisioned with request-queue, token-throughput, TTFT/ITL/e2e-latency percentile, and per-GPU utilization/VRAM/temperature/power panels.

What the session actually taught:

- **vLLM metric names drift across versions.** Two documented names didn't exist on this digest (`gpu_cache_usage_perc` → `kv_cache_usage_perc`; `time_per_output_token_seconds` → `inter_token_latency_seconds`). Consequence recorded as a rule: any dashboard committed to the repo is implicitly pinned to the engine image digest — re-verify names when the digest bumps.
- **Choose the per-step histogram over the per-request average** for inter-token latency: the histogram surfaces decode stalls that a per-request mean smooths away. A deliberate panel-level decision, not a name substitution.
- **The KV-cache gauge reading ~0% under a 20-request burst was expected behavior, verified rather than shrugged off:** a 15 s scrape interval samples instantaneous usage, the burst fit between scrapes, and its peak (~4,600 tokens against a ~95K pool) would have registered ~5% anyway. Direct 1 s polling of `/metrics` during one long generation confirmed the metric moves. Meaningful KV pressure needs sustained long-request load — flagged for the Week 11 load work.
- **DCGM's per-GPU panels double as a topology check:** under TP=2 load on GPUs 0+2, GPUs 1 and 3 should stay flat — a visual catch for a model accidentally landing on the PCIe x1 cards. (The empirical-placement habit, now in dashboard form.)
- Grafana was moved to all-interfaces binding for LAN access — an explicit, temporary posture trade documented with its exit condition (return to loopback once nginx fronts the stack). Prometheus and vLLM stayed loopback-only.

## Not measured this week

No throughput, latency, or load characterization of any kind — the only traffic was smoke requests and a 20-curl burst used to exercise the dashboard. All Week 10 output is infrastructure and configuration.

## Open at week close (as the dailies left them)

- nginx reverse proxy: declared in Compose, never launched. TLS termination, access-log scraping, and re-tightening the Grafana/vLLM host bindings all hung off it and were likewise deferred. No Day 3–5 sessions were journaled; the front-door thread resumes at Week 13 Day 2.
- #39133 still had no assignee or linked PR as of Day 1 (2026-04-13); the digest-pinned stack was explicitly positioned to make the eventual re-test cheap (realized in Week 9 Day 4, 2026-05-17).

## Artifacts produced this week

- The `inference-reference-stack` repository itself: Compose file (4 services + dcgm-exporter added Day 2), streaming-aware `nginx/nginx.conf`, Prometheus scrape config, Grafana provisioning + `vllm-overview.json` starter dashboard, `.env.example`, README with the Triton decision record
- Per-day journals (Days 1–2)

## Carried forward

1. nginx front door + TLS (deferred; picked up in Week 13).
2. Sustained-load testing that would make the KV-pressure and queue panels informative (Week 11's concurrent work).
3. Dashboard-vs-digest compatibility rule — re-verify metric names on every engine bump.
