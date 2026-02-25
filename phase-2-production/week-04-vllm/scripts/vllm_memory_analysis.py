#!/usr/bin/env python3
"""
Week 4 Experiment 4: PagedAttention Memory Analysis
Compare vLLM's memory management vs Week 1 transformers linear model.

Week 1 memory model (transformers):
  Peak Memory = 6.47 GB + 0.2606 MB/token × seq_length × batch_size
  Per-token KV cache: 0.26 MB (measured), 0.33 MB (theoretical)

vLLM startup reported:
  Model weights: 6.016 GiB
  KV cache pool: 13.97 GiB
  Pool capacity: 130,752 tokens
  Max concurrency at 4096 tokens: 31.9x

This experiment measures:
  1. KV cache utilization under varying concurrent load
  2. Effective capacity at different sequence lengths
  3. Per-token memory cost in vLLM vs transformers
  4. How PagedAttention handles mixed sequence lengths

Usage:
  Ensure vLLM is running, then:
  python3 scripts/vllm_memory_analysis.py
"""

import asyncio
import aiohttp
import time
import statistics
import re

VLLM_URL = "http://localhost:8000/v1/completions"
METRICS_URL = "http://localhost:8000/metrics"
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
TEMPERATURE = 0.0

# From vLLM startup log
VLLM_KV_POOL_TOKENS = 130_752
VLLM_KV_POOL_GIB = 13.97
VLLM_MODEL_GIB = 6.016

# Week 1 transformers model
W1_BASE_GB = 6.47
W1_PER_TOKEN_MB = 0.2606


async def fetch_metrics(session):
    """Fetch vLLM Prometheus metrics and parse KV cache utilization."""
    try:
        async with session.get(METRICS_URL) as resp:
            text = await resp.text()
            metrics = {}

            for line in text.split("\n"):
                if line.startswith("#"):
                    continue

                # KV cache usage percentage
                m = re.match(r"vllm:gpu_cache_usage_perc\s+([\d.]+)", line)
                if m:
                    metrics["kv_cache_usage_pct"] = float(m.group(1))

                # Number of running requests
                m = re.match(r"vllm:num_requests_running\s+([\d.]+)", line)
                if m:
                    metrics["requests_running"] = int(float(m.group(1)))

                # Number of waiting requests
                m = re.match(r"vllm:num_requests_waiting\s+([\d.]+)", line)
                if m:
                    metrics["requests_waiting"] = int(float(m.group(1)))

                # Prefix cache hit rate
                m = re.match(r"vllm:gpu_prefix_cache_hit_rate\s+([\d.]+)", line)
                if m:
                    metrics["prefix_cache_hit_rate"] = float(m.group(1))

            return metrics
    except Exception as e:
        return {"error": str(e)}


async def send_and_hold(session, prompt, max_tokens, semaphore, results):
    """Send a request that generates many tokens, keeping KV cache occupied."""
    async with semaphore:
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": TEMPERATURE,
        }
        start = time.perf_counter()
        try:
            async with session.post(VLLM_URL, json=payload) as resp:
                result = await resp.json()
                elapsed = time.perf_counter() - start
                if resp.status == 200:
                    results.append({
                        "prompt_tokens": result["usage"]["prompt_tokens"],
                        "completion_tokens": result["usage"]["completion_tokens"],
                        "total_tokens": result["usage"]["total_tokens"],
                        "latency": elapsed,
                        "success": True,
                    })
                else:
                    results.append({"success": False, "error": str(result)})
        except Exception as e:
            results.append({"success": False, "error": str(e)})


async def measure_kv_utilization(session, num_requests, max_tokens, prompt_base,
                                  sample_interval=0.2):
    """Launch concurrent requests and sample KV cache metrics during execution."""
    # Build prompts of varying length to simulate real workloads
    prompt = prompt_base

    semaphore = asyncio.Semaphore(num_requests)  # All launch at once
    results = []

    # Start requests
    tasks = [
        send_and_hold(session, prompt, max_tokens, semaphore, results)
        for _ in range(num_requests)
    ]
    request_task = asyncio.gather(*tasks)

    # Sample metrics while requests are running
    metric_samples = []
    sample_start = time.perf_counter()

    while not request_task.done():
        metrics = await fetch_metrics(session)
        metrics["elapsed"] = time.perf_counter() - sample_start
        metric_samples.append(metrics)
        await asyncio.sleep(sample_interval)

    await request_task

    return results, metric_samples


async def main():
    print("=" * 90)
    print("WEEK 4 EXPERIMENT 4: PAGEDATTENTION MEMORY ANALYSIS")
    print("=" * 90)
    print()
    print(f"vLLM KV cache pool: {VLLM_KV_POOL_GIB:.2f} GiB ({VLLM_KV_POOL_TOKENS:,} tokens)")
    print(f"vLLM model weights:  {VLLM_MODEL_GIB:.3f} GiB")
    print(f"Week 1 per-token:    {W1_PER_TOKEN_MB:.4f} MB (transformers measured)")
    print()

    # ── Part A: Per-token memory cost comparison ──
    print("=" * 90)
    print("PART A: PER-TOKEN MEMORY COST COMPARISON")
    print("=" * 90)
    print()

    vllm_per_token_mb = (VLLM_KV_POOL_GIB * 1024) / VLLM_KV_POOL_TOKENS
    theoretical_per_token_mb = (2 * 28 * 24 * 128 * 2) / (1024 * 1024)  # Llama 3.2 3B

    print(f"  Theoretical (architecture):  {theoretical_per_token_mb:.4f} MB/token")
    print(f"  Transformers (Week 1 measured): {W1_PER_TOKEN_MB:.4f} MB/token")
    print(f"  vLLM (pool / capacity):      {vllm_per_token_mb:.4f} MB/token")
    print(f"  vLLM vs Transformers:        {vllm_per_token_mb/W1_PER_TOKEN_MB:.2f}x")
    print(f"  vLLM vs Theoretical:         {vllm_per_token_mb/theoretical_per_token_mb:.2f}x")
    print()

    # ── Part B: Capacity at different sequence lengths ──
    print("=" * 90)
    print("PART B: CAPACITY AT DIFFERENT SEQUENCE LENGTHS")
    print("=" * 90)
    print()
    print(f"{'Seq Length':>10} | {'vLLM Max Conc':>14} | {'TF Max Batch':>14} | "
          f"{'TF Memory/Req':>14} | {'vLLM Advantage':>14}")
    print("-" * 80)

    seq_lengths = [50, 100, 256, 500, 1024, 2048, 4096]

    for seq_len in seq_lengths:
        # vLLM: pool capacity / tokens per request
        vllm_max = VLLM_KV_POOL_TOKENS / seq_len

        # Transformers: (24 GB - base) / per_request_memory
        tf_per_req_gb = (W1_PER_TOKEN_MB * seq_len) / 1024
        tf_available_gb = 24.0 - W1_BASE_GB - 2.5  # 2.5 GB safety margin
        tf_max = tf_available_gb / tf_per_req_gb if tf_per_req_gb > 0 else 0

        advantage = vllm_max / tf_max if tf_max > 0 else float('inf')

        print(f"{seq_len:>10} | {vllm_max:>14.1f} | {tf_max:>14.1f} | "
              f"{tf_per_req_gb*1024:>12.1f} MB | {advantage:>13.2f}x")

    print()

    # ── Part C: Live KV cache utilization under load ──
    print("=" * 90)
    print("PART C: LIVE KV CACHE UTILIZATION UNDER LOAD")
    print("=" * 90)
    print()

    timeout = aiohttp.ClientTimeout(total=300)
    connector = aiohttp.TCPConnector(limit=0)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Warmup
        print("Warming up...")
        warmup_results = []
        sem = asyncio.Semaphore(5)
        tasks = [
            send_and_hold(session, "Hello", 20, sem, warmup_results)
            for _ in range(5)
        ]
        await asyncio.gather(*tasks)
        print("Warmup complete.\n")

        # Baseline metrics (idle)
        idle_metrics = await fetch_metrics(session)
        print(f"Idle KV cache usage: {idle_metrics.get('kv_cache_usage_pct', 'N/A')}")
        print()

        # Test scenarios: (description, num_requests, max_tokens_per_request)
        scenarios = [
            ("10 requests × 50 tokens",   10,  50),
            ("25 requests × 50 tokens",   25,  50),
            ("10 requests × 200 tokens",  10, 200),
            ("25 requests × 200 tokens",  25, 200),
            ("10 requests × 500 tokens",  10, 500),
            ("25 requests × 500 tokens",  25, 500),
            ("30 requests × 500 tokens",  30, 500),
            ("10 requests × 2000 tokens", 10, 2000),
            ("25 requests × 2000 tokens", 25, 2000),
        ]

        prompt = ("Explain in great detail the following topic, covering all aspects "
                  "including history, current state, and future implications: "
                  "distributed computing systems and their role in modern infrastructure")

        print(f"{'Scenario':<30} | {'Peak KV%':>8} | {'Tokens Used':>12} | "
              f"{'Sys tok/s':>10} | {'Success':>8} | {'Wait':>5}")
        print("-" * 90)

        for desc, num_req, max_tok in scenarios:
            results, samples = await measure_kv_utilization(
                session, num_req, max_tok, prompt, sample_interval=0.3
            )

            successful = [r for r in results if r["success"]]
            failed = [r for r in results if not r["success"]]

            # Peak KV cache usage during this scenario
            kv_samples = [s.get("kv_cache_usage_pct", 0) for s in samples]
            peak_kv = max(kv_samples) if kv_samples else 0
            peak_kv_tokens = int(peak_kv * VLLM_KV_POOL_TOKENS)

            # Waiting requests observed
            wait_samples = [s.get("requests_waiting", 0) for s in samples]
            peak_wait = max(wait_samples) if wait_samples else 0

            # Throughput
            if successful:
                total_tokens = sum(r["completion_tokens"] for r in successful)
                total_time = max(r["latency"] for r in successful)
                sys_tps = total_tokens / total_time if total_time > 0 else 0
            else:
                sys_tps = 0

            print(f"{desc:<30} | {peak_kv:>7.1%} | {peak_kv_tokens:>12,} | "
                  f"{sys_tps:>10,.1f} | "
                  f"{len(successful):>3}/{num_req:<3} | {peak_wait:>5}")

            # Brief pause between scenarios for KV cache to clear
            await asyncio.sleep(2)

        # ── Part D: Mixed sequence length efficiency ──
        print()
        print("=" * 90)
        print("PART D: MIXED SEQUENCE LENGTH EFFICIENCY")
        print("=" * 90)
        print()
        print("Testing: 30 concurrent requests with mixed generation lengths")
        print("  10 × short (20 tokens) + 10 × medium (200 tokens) + 10 × long (500 tokens)")
        print()

        mixed_results = []
        mixed_samples = []
        sem_mixed = asyncio.Semaphore(30)
        sample_start = time.perf_counter()

        # Build mixed workload
        mixed_tasks = []
        for i in range(10):
            mixed_tasks.append(
                send_and_hold(session, "Answer briefly:", 20, sem_mixed, mixed_results))
        for i in range(10):
            mixed_tasks.append(
                send_and_hold(session, "Explain in detail:", 200, sem_mixed, mixed_results))
        for i in range(10):
            mixed_tasks.append(
                send_and_hold(session, "Write a comprehensive analysis of:", 500,
                             sem_mixed, mixed_results))

        gather_task = asyncio.gather(*mixed_tasks)

        while not gather_task.done():
            metrics = await fetch_metrics(session)
            metrics["elapsed"] = time.perf_counter() - sample_start
            mixed_samples.append(metrics)
            await asyncio.sleep(0.3)

        await gather_task

        # Analyze by group
        short = [r for r in mixed_results if r["success"] and r["completion_tokens"] <= 25]
        medium = [r for r in mixed_results if r["success"] and 25 < r["completion_tokens"] <= 250]
        long = [r for r in mixed_results if r["success"] and r["completion_tokens"] > 250]

        for label, group in [("Short (≤25 tok)", short),
                              ("Medium (26-250 tok)", medium),
                              ("Long (>250 tok)", long)]:
            if group:
                lats = [r["latency"] for r in group]
                tps = [r["completion_tokens"] / r["latency"] for r in group]
                avg_tok = statistics.mean(r["completion_tokens"] for r in group)
                print(f"  {label:.<25} count={len(group):>3}, "
                      f"avg_gen={avg_tok:>6.1f}t, "
                      f"tok/s={statistics.mean(tps):>6.1f}, "
                      f"lat_p50={statistics.median(lats):>6.3f}s, "
                      f"lat_p95={sorted(lats)[int(0.95*len(lats)):][0] if len(lats) > 1 else lats[0]:>6.3f}s")

        # Key observation about continuous batching
        if short and long:
            short_lat = statistics.mean(r["latency"] for r in short)
            long_lat = statistics.mean(r["latency"] for r in long)
            print()
            print(f"  Short requests completed in {short_lat:.2f}s avg vs "
                  f"long at {long_lat:.2f}s")
            print(f"  Without continuous batching, ALL requests would wait "
                  f"for the longest ({long_lat:.2f}s)")
            print(f"  Continuous batching saved short requests "
                  f"{long_lat - short_lat:.2f}s ({(1-short_lat/long_lat)*100:.0f}% reduction)")

    print()
    print("=" * 90)
    print("EXPERIMENT 4 COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    asyncio.run(main())
