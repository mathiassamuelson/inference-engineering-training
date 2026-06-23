# Week 10 Day 2 — Prometheus, Grafana, and DCGM

## Objective

Stand up the observability layer alongside the Day 1 vLLM deployment: Prometheus scraping the vLLM `/metrics` endpoint, Grafana visualizing via a starter dashboard, and (stretch) `dcgm-exporter` providing GPU-level metrics to complement the application-level ones vLLM exposes.

## Stack state at end of session

Four services running under Docker Compose on the `irs` bridge network:

| Service        | Image                                  | Host binding          | Internal port |
|----------------|----------------------------------------|-----------------------|---------------|
| vllm           | `vllm/vllm-openai` (digest-pinned)     | `127.0.0.1:8000`      | 8000          |
| prometheus     | `prom/prometheus:v2.54.1`              | `127.0.0.1:9090`      | 9090          |
| grafana        | `grafana/grafana:11.3.0`               | `0.0.0.0:3000`        | 3000          |
| dcgm-exporter  | `nvidia/dcgm-exporter:3.3.9-3.6.1-ubuntu22.04` | `127.0.0.1:9400` | 9400          |

nginx remains declared but not launched; Day 3–4 scope.

## What moved

### vLLM metric name drift

The starter dashboard I drafted against vLLM's documented metric names hit two misses on this image digest. Resolved by enumerating actual exposed metrics and patching the dashboard PromQL:

| Dashboard reference                            | Actually exposed in this build                  |
|------------------------------------------------|-------------------------------------------------|
| `vllm:gpu_cache_usage_perc`                    | `vllm:kv_cache_usage_perc`                      |
| `vllm:time_per_output_token_seconds_bucket`    | `vllm:inter_token_latency_seconds_bucket`       |
| `vllm:time_to_first_token_seconds_bucket`      | (matched)                                       |
| `vllm:e2e_request_latency_seconds_bucket`      | (matched)                                       |

The `time_per_output_token` → `inter_token_latency` choice wasn't arbitrary: this build does expose a `vllm:request_time_per_output_token_seconds` metric (a per-request average), but `inter_token_latency_seconds` is the per-step histogram, which is the better surfacer of decode stalls under load. The per-request average would smooth those stalls over the full generation.

Worth noting for the repo README: vLLM's metric naming has drifted across versions, and any dashboard committed to the repo is implicitly pinned to the vLLM image digest. Going to add a "Dashboard compatibility" note when the observability README gets written.

### Grafana host binding

Initial Compose config published Grafana on `127.0.0.1:3000`. For a reference stack that operators may want to hit from a laptop on the LAN, loopback-only is wrong. Swapped the `ports:` entry from `"127.0.0.1:3000:3000"` to `"3000:3000"` to bind all interfaces.

This trades one security posture for another. The `GRAFANA_ADMIN_PASSWORD` in `.env` is now the only thing between the LAN and the dashboards; acceptable on a trusted network, not for production. Long-term: Day 3–4 introduces nginx as the sole public-facing service with TLS, at which point Grafana can go back to loopback-only and nginx takes over LAN/WAN access.

Kept Prometheus (`9090`) and vLLM (`8000`) on loopback-only. Prometheus in particular isn't something I want casually reachable — debug access via SSH tunnel is fine.

### Starter dashboard

Committed at `observability/grafana/dashboards/vllm-overview.json`, auto-provisioned via the existing `grafana/provisioning/` wiring from Day 1. Panels:

- Top row (stats): Requests Running, Requests Waiting, KV Cache Utilization (gauge)
- Request Queue State (running + waiting, time series)
- Token Throughput (prompt tok/s, generation tok/s via `rate(..._total[1m])`)
- TTFT percentiles p50/p95/p99 (histogram_quantile)
- Inter-Token Latency (TPOT) percentiles
- End-to-End Request Latency percentiles
- GPU row: utilization, VRAM used, temperature, power (all per-GPU, legend `GPU {{gpu}}`)

Used a `${DS_PROMETHEUS}` datasource variable rather than hardcoding the provisioned UID. Dashboard is portable across stacks if the Prometheus datasource name matches.

### DCGM exporter

Added as a fifth Compose service with `capabilities: [gpu, utility]` via the Compose V2 device reservation syntax. First target query returned `result: []`, which was a scrape-timing race rather than a real failure — Prometheus hadn't completed its first DCGM scrape by the time I queried it. Retry showed the target UP and metrics flowing.

Metric labels exposed per GPU include `gpu` (0–3), `UUID`, `pci_bus_id`, and `modelName`. The `gpu` label is the natural grouping key and is used directly in panel legends.

The dashboard's GPU panels will show a useful topology signal during load: with the 26B A4B MoE loaded TP=2 across GPUs 0 and 2 (the NVLink pair), GPUs 1 and 3 should stay flat under inference load. That visual confirmation of the deployment topology is itself useful — it's the kind of thing that catches a misconfiguration where the model accidentally lands on the PCIe x1 GPUs.

### KV cache gauge — zero-reading was expected

First load test (20 concurrent short curl requests, `max_tokens=200`) left the KV cache panel stuck at ~0%. Not a bug. Two effects compounding:

1. Prometheus scrapes every 15s; the gauge samples instantaneous usage at scrape time. A burst that completes in a few seconds passes entirely between scrapes.
2. Peak usage during that burst is tiny anyway: 20 × ~230 tokens = ~4,600 tokens against a pool that holds ~95k tokens/GPU. Even if a scrape caught the peak, it'd register ~5%.

Confirmed the metric works by polling `/metrics` directly at 1-second intervals during a single long (2000-token) generation. Non-zero values appear as expected. The dashboard panel will show meaningful KV pressure under sustained, longer-request load — something the starter load generator doesn't produce, but a proper load test in Week 11 will.

## Carry-forward notes

- **Metric names are version-bound.** Any dashboard in the repo is implicitly tied to the vLLM image digest. When the digest bumps (e.g., when [vllm-project/vllm#39133](https://github.com/vllm-project/vllm/issues/39133) fixes land), re-verify metric names before trusting the dashboard.
- **`DCGM_FI_DEV_FB_USED` is reported in MiB.** The dashboard panel multiplies by `1024 * 1024` inline so Grafana's `decbytes` unit renders GiB correctly. Alternative would be the `deczbytes` family, but inline conversion keeps the panel config explicit.
- **Grafana picks up dashboard JSON changes within the 30s file-update interval.** Restart only needed if you want it immediate.
- **DCGM scrape race:** first query after `docker compose up -d dcgm-exporter` may return empty results. Wait one scrape interval (15s) and retry.
- **The bridge network is doing real work.** Prometheus resolves `vllm:8000`, `dcgm-exporter:9400`, etc. via Compose's internal DNS. Day 1's decision to move off `--network host` pays off here.

## What's deferred to Day 3–4

- nginx reverse proxy (already declared in Compose, deliberately not launched).
- TLS termination at nginx.
- nginx access log scraping (requires nginx up first).
- Tightening vLLM and Grafana host bindings back to loopback once nginx fronts them.

## References

- [vLLM Production Metrics docs](https://docs.vllm.ai/en/stable/serving/metrics.html)
- [Grafana provisioning docs](https://grafana.com/docs/grafana/latest/administration/provisioning/)
- [DCGM exporter metrics reference](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html)
- Upstream KV-sizing issue: [vllm-project/vllm#39133](https://github.com/vllm-project/vllm/issues/39133)
