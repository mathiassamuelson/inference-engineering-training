# Week 10, Day 1 — vLLM Compose scaffold

**Date:** 2026-04-13

## Objective

Stand up the skeleton of the new public deployment repository (`inference-reference-stack`) and get the inference engine running under Docker Compose. Subsequent days add observability and TLS/reverse proxy work on top.

## What landed

A new public GitHub repository, `inference-reference-stack`, initialized with:

- Apache 2.0 license
- `.gitignore` tuned for Docker, Python, HF caches, TLS material, and service data volumes
- README with project pitch, architecture diagram, component list, and roadmap
- `docker-compose.yml` declaring four services: `vllm`, `nginx`, `prometheus`, `grafana`
- `nginx/nginx.conf` with streaming-aware reverse proxy config (buffering disabled on `/v1/chat/completions` and `/v1/completions`)
- `observability/prometheus/prometheus.yml` scraping the inference engine's `/metrics` endpoint
- `observability/grafana/provisioning/` for automatic datasource wiring
- `.env.example` template for `HF_CACHE_DIR`, `HF_TOKEN`, and `GRAFANA_ADMIN_PASSWORD`

Only the `vllm` service was actually started today. nginx, Prometheus, and Grafana are declared in Compose but not yet launched — that's Day 2.

## The Triton pivot

Original Week 10 plan called for Triton Inference Server fronting vLLM as the serving layer. That got abandoned at Day 1 for a concrete technical reason:

1. Current NGC Triton vLLM image is `nvcr.io/nvidia/tritonserver:26.02-vllm-python-py3`.
2. That image bundles `vllm 0.15.1+nv26.2`.
3. Week 9's Gemma 4 AWQ-INT4 work required `vllm 0.18.2rc1.dev73` specifically, via the `compressed-tensors` path with Marlin INT4 kernels. 0.15.1 predates that by three minor releases and is at-best risky, at-worst a hard incompatibility for this model.

Three options were on the table: test whether 0.15 happens to work, build a custom Dockerfile layering newer vLLM over the NGC base, or drop Triton and use standalone vLLM's built-in OpenAI-compatible server. Settled on the last one.

The rationale that tipped it: the architectural value of this repo — reverse proxy, observability, eventual gateway and metering — lives above the inference engine. Whether the engine is Triton or vanilla vLLM doesn't change any of that. Meanwhile, dropping Triton preserves the exact Week 9 image for the pending [vllm-project/vllm#39133](https://github.com/vllm-project/vllm/issues/39133) before/after comparison — which is a concrete asset that a Triton detour would have muddied.

The vLLM image is pinned by full digest (`@sha256:0cb12dc9...`) rather than tag, so when #39133 lands a fix we swap the digest in isolation and attribute any performance change unambiguously to the fix rather than to unrelated vLLM version drift.

README includes a "Why not Triton?" subsection documenting the decision in-repo rather than leaving it in commit history only. Triton is not permanently off the table — it comes back if and when multi-model serving or ensemble workflows justify the added layer.

## The Compose V2 detour

First `docker compose up` revealed Compose V2 wasn't installed. The host has Ubuntu's `docker.io` package (28.2.2), not Docker Inc.'s `docker-ce`. That distinction matters because `docker-compose-plugin` (Docker Inc.'s V2 plugin package) isn't in Ubuntu's default repos; only `docker-compose-v2` is. `sudo apt-get install -y docker-compose-v2` resolved it cleanly. Worth remembering: `docker.io` and `docker-ce` have parallel-but-different plugin package names.

## End state

`docker compose up vllm` brings up the engine cleanly, loads Gemma 4 26B MoE across GPUs 0 and 2 via TP=2, and serves requests:

- `GET /v1/models` — returns the model listing with `id: cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`, `max_model_len: 16384`.
- `POST /v1/chat/completions` — round-trips a "pong" reply in 2 completion tokens.
- `GET /metrics` — returns Prometheus metrics (Python process and vLLM-prefixed).

The launch command in `docker-compose.yml` mirrors the exact Week 9 `docker run` arguments: positional model path, `--tensor-parallel-size 2`, `--max-model-len 16384`, `--gpu-memory-utilization 0.90`, `--limit-mm-per-prompt '{"image":0,"audio":0}'`, explicit `--host 0.0.0.0 --port 8000`. Port 8000 is bound to `127.0.0.1` only on the host; nginx will be the external ingress once it comes up. Host `--network host` was deliberately not replicated — Compose's internal bridge network is needed so nginx can resolve `vllm:8000` via service DNS.

## Notes for later

- **Image digest is the reproducibility anchor.** If someone needs to run the #39133 re-test and the vLLM image has been retagged upstream in the meantime, the digest pin keeps them honest. If the digest pin ever starts failing (image purged from Docker Hub, say), that's a signal to check upstream rather than silently drift.
- **Grafana dashboard directory is empty.** The provisioning file points at `/var/lib/grafana/dashboards` and will pick up any JSON dropped there on Grafana restart. Day 2 populates it.
- **vLLM metrics live at `:8000/metrics`**, same port as the API. Different from Triton's dedicated `:8002`. Prometheus scrape config reflects this; worth not confusing next time I reach for a Triton-shaped mental model.
- **#39133 status:** still no assignee or linked PR as of today. Watch item.

## What's deferred

- Prometheus and Grafana containers (configured but not yet launched) — Day 2
- nginx reverse proxy actually fronting vLLM — Day 2–3
- HTTPS / TLS termination — Day 3–4
- End-of-week writeup — Day 5

## Key references

- Repository: `inference-reference-stack` (public)
- Model: [`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`](https://huggingface.co/cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit)
- Image digest: `vllm/vllm-openai@sha256:0cb12dc964e1dace0a78aecd8905461d851b135db0690726f08550f7c4922834`
- NGC Triton catalog: [nvcr.io/nvidia/tritonserver](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)
- vLLM KV-sizing issue: [vllm-project/vllm#39133](https://github.com/vllm-project/vllm/issues/39133)
