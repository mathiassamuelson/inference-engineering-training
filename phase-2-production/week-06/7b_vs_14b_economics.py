#!/usr/bin/env python3
"""
Week 6 Experiment 2: Single-GPU 7B vs Multi-GPU 14B Serving Economics
Head-to-head comparison answering: "When does upgrading model size justify extra GPUs?"

Methodology:
  - Mistral 7B on 1x RTX 3090 (this benchmark)
  - Qwen 2.5 14B on 2x RTX 3090 TP=2 (data from Experiment 1)
  - Same concurrency levels, same generation length, same prompt
  - Compare: throughput, latency, cost-per-token, tokens-per-GPU

Usage:
  1. Start Mistral 7B server on GPU 0:
     CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
         --model mistralai/Mistral-7B-Instruct-v0.3 --dtype float16 \
         --max-model-len 4096 --port 8000 --disable-log-requests

  2. Run benchmark:
     python3 7b_vs_14b_economics.py
"""

import asyncio
import aiohttp
import time
import json
import statistics
import sys

# ── Configuration ──────────────────────────────────────────────────
VLLM_URL = "http://localhost:8000/v1/completions"
MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
PROMPT = "Explain the concept of GPU memory bandwidth and its impact on"
MAX_TOKENS = 50
TEMPERATURE = 0.0
WARMUP_REQUESTS = 5
NUM_ITERATIONS = 3

# Same concurrency levels as Experiment 1 for direct comparison
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# ── Experiment 1 results: Qwen 2.5 14B, TP=2, 2x GPU ─────────────
QWEN_14B_RESULTS = {
    1:   {"total":  38.6, "per_sample": 38.6, "latency_mean": 1.295, "latency_p95": 1.295},
    2:   {"total":  68.1, "per_sample": 34.1, "latency_mean": 1.455, "latency_p95": 1.468},
    4:   {"total": 112.0, "per_sample": 28.0, "latency_mean": 1.777, "latency_p95": 1.785},
    8:   {"total": 165.1, "per_sample": 20.6, "latency_mean": 2.417, "latency_p95": 2.423},
    16:  {"total": 218.3, "per_sample": 13.6, "latency_mean": 3.660, "latency_p95": 3.665},
    32:  {"total": 267.3, "per_sample":  8.4, "latency_mean": 5.982, "latency_p95": 5.986},
    64:  {"total": 303.6, "per_sample":  4.7, "latency_mean": 10.536, "latency_p95": 10.539},
    128: {"total": 312.1, "per_sample":  2.4, "latency_mean": 20.498, "latency_p95": 20.502},
    256: {"total": 316.5, "per_sample":  1.2, "latency_mean": 40.280, "latency_p95": 40.437},
}

# ── Hardware cost assumptions ──────────────────────────────────────
# RTX 3090 TDP: 350W, electricity ~$0.12/kWh
GPU_COST_USD = 1500        # purchase price per GPU
GPU_TDP_WATTS = 350        # thermal design power
ELECTRICITY_PER_KWH = 0.12
# Cloud equivalent: ~$1.00/hr per GPU (rough A10G/L4 equivalent)
CLOUD_COST_PER_GPU_HR = 1.00


async def send_request(session, request_id):
    """Send a single completion request and return timing + token count."""
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    start = time.perf_counter()
    try:
        async with session.post(VLLM_URL, json=payload,
                                timeout=aiohttp.ClientTimeout(total=120)) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                return {"success": False, "error": f"HTTP {resp.status}: {error_text[:200]}",
                        "latency": time.perf_counter() - start, "tokens": 0}
            data = await resp.json()
            elapsed = time.perf_counter() - start
            tokens = data["usage"]["completion_tokens"]
            return {"success": True, "latency": elapsed, "tokens": tokens,
                    "request_id": request_id}
    except Exception as e:
        return {"success": False, "error": str(e)[:200],
                "latency": time.perf_counter() - start, "tokens": 0}


async def run_concurrent_batch(session, num_requests):
    """Fire num_requests simultaneously, measure wall-clock time."""
    tasks = [send_request(session, i) for i in range(num_requests)]
    wall_start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    wall_elapsed = time.perf_counter() - wall_start
    return results, wall_elapsed


async def benchmark_concurrency(session, concurrency, num_iterations):
    """Run multiple iterations at a given concurrency level."""
    iteration_results = []
    for iteration in range(num_iterations):
        results, wall_time = await run_concurrent_batch(session, concurrency)
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        if not successful:
            print(f"    Iteration {iteration+1}: ALL REQUESTS FAILED")
            if failed:
                print(f"      Error: {failed[0]['error']}")
            continue

        total_tokens = sum(r["tokens"] for r in successful)
        latencies = [r["latency"] for r in successful]
        total_throughput = total_tokens / wall_time
        per_sample_throughput = total_tokens / (wall_time * len(successful))

        iteration_results.append({
            "wall_time": wall_time,
            "total_tokens": total_tokens,
            "successful": len(successful),
            "failed": len(failed),
            "total_throughput": total_throughput,
            "per_sample_throughput": per_sample_throughput,
            "latency_mean": statistics.mean(latencies),
            "latency_p50": statistics.median(latencies),
            "latency_p95": sorted(latencies)[int(0.95 * len(latencies))]
                          if len(latencies) > 1 else latencies[0],
            "latency_p99": sorted(latencies)[int(0.99 * len(latencies))]
                          if len(latencies) > 1 else latencies[0],
        })
    return iteration_results


def print_economics(mistral_results):
    """Print the side-by-side economics comparison."""

    print()
    print("=" * 130)
    print("ECONOMICS COMPARISON: MISTRAL 7B (1 GPU) vs QWEN 14B (2 GPU, TP=2)")
    print("=" * 130)

    # ── Throughput comparison ──
    print()
    print("THROUGHPUT (tok/s)")
    print("-" * 130)
    print(f"{'Conc':>6} | {'7B Total':>10} {'7B/Samp':>9} | "
          f"{'14B Total':>10} {'14B/Samp':>9} | "
          f"{'Total Ratio':>11} {'Samp Ratio':>11} | "
          f"{'7B Lat p95':>10} {'14B Lat p95':>11}")
    print("-" * 130)

    for conc in CONCURRENCY_LEVELS:
        m = mistral_results.get(conc)
        q = QWEN_14B_RESULTS.get(conc)
        if not m:
            print(f"{conc:>6} | {'FAILED':>10}")
            continue

        total_ratio = m["total_throughput"] / q["total"] if q else 0
        samp_ratio = m["per_sample_throughput"] / q["per_sample"] if q else 0

        print(f"{conc:>6} | "
              f"{m['total_throughput']:>8,.1f}   "
              f"{m['per_sample_throughput']:>7.1f}   | "
              f"{q['total']:>8,.1f}   "
              f"{q['per_sample']:>7.1f}   | "
              f"{total_ratio:>9.2f}x   "
              f"{samp_ratio:>9.2f}x   | "
              f"{m['latency_p95']:>9.3f}s "
              f"{q['latency_p95']:>10.3f}s")

    # ── Cost efficiency ──
    print()
    print("=" * 130)
    print("COST EFFICIENCY")
    print("=" * 130)

    # Find peak throughput for each
    m_peak_conc = max(mistral_results, key=lambda c: mistral_results[c]["total_throughput"])
    m_peak = mistral_results[m_peak_conc]["total_throughput"]
    q_peak_conc = max(QWEN_14B_RESULTS, key=lambda c: QWEN_14B_RESULTS[c]["total"])
    q_peak = QWEN_14B_RESULTS[q_peak_conc]["total"]

    print()
    print(f"  {'Metric':<40} {'Mistral 7B (1 GPU)':>20} {'Qwen 14B (2 GPU)':>20}")
    print(f"  {'-'*40} {'-'*20} {'-'*20}")
    print(f"  {'GPUs used':<40} {'1':>20} {'2':>20}")
    print(f"  {'Model parameters':<40} {'7B':>20} {'14B':>20}")
    print(f"  {'Single-request tok/s':<40} "
          f"{mistral_results[1]['total_throughput']:>18.1f}  "
          f"{QWEN_14B_RESULTS[1]['total']:>18.1f}  ")
    print(f"  {'Peak throughput (tok/s)':<40} "
          f"{m_peak:>18,.1f}  "
          f"{q_peak:>18,.1f}  ")
    print(f"  {'Peak throughput per GPU (tok/s)':<40} "
          f"{m_peak:>18,.1f}  "
          f"{q_peak/2:>18,.1f}  ")

    # Tokens per dollar (cloud)
    m_tokens_per_dollar = m_peak / CLOUD_COST_PER_GPU_HR * 3600  # tokens per $1
    q_tokens_per_dollar = q_peak / (2 * CLOUD_COST_PER_GPU_HR) * 3600

    print(f"  {'Cloud cost ($/hr)':<40} "
          f"{'$%.2f' % (1 * CLOUD_COST_PER_GPU_HR):>20} "
          f"{'$%.2f' % (2 * CLOUD_COST_PER_GPU_HR):>20}")
    print(f"  {'Peak tokens per $1 (cloud)':<40} "
          f"{m_tokens_per_dollar:>18,.0f}  "
          f"{q_tokens_per_dollar:>18,.0f}  ")
    print(f"  {'Cost ratio (14B/7B per token)':<40} "
          f"{'1.00x (baseline)':>20} "
          f"{m_tokens_per_dollar/q_tokens_per_dollar:>.2f}x  ".rjust(21))

    # Power cost
    m_watts = 1 * GPU_TDP_WATTS
    q_watts = 2 * GPU_TDP_WATTS
    m_cost_per_mtok = (m_watts / 1000 * ELECTRICITY_PER_KWH) / (m_peak * 3.6)  # $ per M tokens
    q_cost_per_mtok = (q_watts / 1000 * ELECTRICITY_PER_KWH) / (q_peak * 3.6)

    print()
    print(f"  {'Power draw (TDP, watts)':<40} "
          f"{m_watts:>18}W  "
          f"{q_watts:>18}W  ")
    print(f"  {'Electricity cost per M tokens':<40} "
          f"{'$%.4f' % m_cost_per_mtok:>20} "
          f"{'$%.4f' % q_cost_per_mtok:>20}")

    # ── SLA analysis ──
    print()
    print("=" * 130)
    print("SLA ANALYSIS: MAX CONCURRENT USERS BY LATENCY TARGET")
    print("=" * 130)
    print()

    sla_targets = [1.0, 2.0, 5.0, 10.0]

    print(f"  {'p95 Target':<15} {'7B Users':>10} {'7B tok/s':>10} | "
          f"{'14B Users':>10} {'14B tok/s':>10} | {'Quality':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10}   {'-'*10} {'-'*10}   {'-'*10}")

    for target in sla_targets:
        # Find max concurrency where p95 <= target
        m_max = 0
        m_tp = 0
        for conc in CONCURRENCY_LEVELS:
            if conc in mistral_results and mistral_results[conc]["latency_p95"] <= target:
                m_max = conc
                m_tp = mistral_results[conc]["total_throughput"]

        q_max = 0
        q_tp = 0
        for conc in CONCURRENCY_LEVELS:
            if conc in QWEN_14B_RESULTS and QWEN_14B_RESULTS[conc]["latency_p95"] <= target:
                q_max = conc
                q_tp = QWEN_14B_RESULTS[conc]["total"]

        print(f"  p95 < {target:>4.1f}s    "
              f"{m_max:>10} {m_tp:>8.1f}   | "
              f"{q_max:>10} {q_tp:>8.1f}   | "
              f"{'14B >> 7B':>10}")

    # ── Decision framework ──
    print()
    print("=" * 130)
    print("DECISION FRAMEWORK")
    print("=" * 130)
    print()

    m_single = mistral_results[1]["total_throughput"]
    q_single = QWEN_14B_RESULTS[1]["total"]
    latency_ratio = QWEN_14B_RESULTS[1]["latency_mean"] / mistral_results[1]["latency_mean"]

    print(f"  Question: Is the 14B model worth 2x the GPU cost?")
    print()
    print(f"  Latency penalty:     {latency_ratio:.2f}x slower per request")
    print(f"  Throughput penalty:   {m_peak/q_peak:.2f}x fewer total tokens")
    print(f"  Per-GPU efficiency:  {m_peak:.0f} tok/s/GPU (7B) vs "
          f"{q_peak/2:.0f} tok/s/GPU (14B)")
    print(f"  Cost per token:      {m_tokens_per_dollar/q_tokens_per_dollar:.2f}x more expensive (14B)")
    print()
    print(f"  USE 7B WHEN: Latency-sensitive, high-throughput, cost-constrained,")
    print(f"               simple tasks (summarization, extraction, classification)")
    print()
    print(f"  USE 14B WHEN: Quality-critical, reasoning-heavy, acceptable latency,")
    print(f"                tasks where wrong answers cost more than GPU time")
    print()


async def main():
    print("=" * 100)
    print("WEEK 6 EXPERIMENT 2: MISTRAL 7B (1 GPU) THROUGHPUT BENCHMARK")
    print("=" * 100)
    print(f"Model:           {MODEL}")
    print(f"GPUs:            1x RTX 3090 (GPU 0)")
    print(f"Generation:      {MAX_TOKENS} tokens per request")
    print(f"Warmup:          {WARMUP_REQUESTS} requests")
    print(f"Iterations:      {NUM_ITERATIONS} per concurrency level")
    print(f"Concurrency:     {CONCURRENCY_LEVELS}")
    print()

    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    async with aiohttp.ClientSession(connector=connector) as session:

        # ── Health check ──
        try:
            async with session.get("http://localhost:8000/health",
                                   timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    print("ERROR: Server not healthy")
                    return
        except Exception as e:
            print(f"ERROR: Cannot reach server: {e}")
            return

        print("Server health check: OK")
        print()

        # ── Warmup ──
        print(f"Warming up with {WARMUP_REQUESTS} requests...")
        for i in range(WARMUP_REQUESTS):
            result = await send_request(session, f"warmup_{i}")
            if not result["success"]:
                print(f"  Warmup request {i} failed: {result['error']}")
            else:
                print(f"  Warmup {i+1}: {result['tokens']} tokens in "
                      f"{result['latency']:.3f}s")
        print()

        # ── Benchmark ──
        mistral_results = {}

        for conc in CONCURRENCY_LEVELS:
            print(f"── Concurrency {conc} ──")
            results = await benchmark_concurrency(session, conc, NUM_ITERATIONS)

            if not results:
                print(f"  All iterations failed at concurrency {conc}")
                print()
                continue

            avg_total_tp = statistics.mean(r["total_throughput"] for r in results)
            avg_per_sample = statistics.mean(r["per_sample_throughput"] for r in results)
            avg_latency = statistics.mean(r["latency_mean"] for r in results)
            avg_p95 = statistics.mean(r["latency_p95"] for r in results)

            mistral_results[conc] = {
                "total_throughput": avg_total_tp,
                "per_sample_throughput": avg_per_sample,
                "latency_mean": avg_latency,
                "latency_p95": avg_p95,
                "iterations": results,
            }

            print(f"  Total: {avg_total_tp:,.1f} tok/s | "
                  f"Per-sample: {avg_per_sample:.1f} tok/s | "
                  f"Latency mean: {avg_latency:.3f}s p95: {avg_p95:.3f}s")
            print()

        # ── Print economics comparison ──
        print_economics(mistral_results)


if __name__ == "__main__":
    asyncio.run(main())
