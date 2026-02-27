#!/usr/bin/env python3
"""
Week 6 Experiment 1: Qwen 2.5 14B (TP=2) Throughput Benchmark
Compare against Week 4/5 baselines to measure how vLLM's advantage scales with model size.

Baselines:
  Week 1 (transformers, Llama 3.2 3B, single GPU):
    Batch 1:    84 tok/s total
    Peak:       ~5,000 tok/s (batch 1200)

  Week 4 (vLLM, Llama 3.2 3B, single GPU):
    Batch 1:    106 tok/s total
    Peak:       ~6,100 tok/s

  Week 5 (vLLM, Mistral 7B, single GPU):
    [To be filled from your Week 5 data]

  Week 5 (vLLM, Llama 3.2 3B, 4-GPU data parallel):
    System peak: 7.12x over transformers baseline

Usage:
  1. Start Qwen 2.5 14B server on GPUs 0,1:
     CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server \
         --model Qwen/Qwen2.5-14B-Instruct --tensor-parallel-size 2 \
         --dtype float16 --max-model-len 4096 --port 8000 --disable-log-requests

  2. Run benchmark:
     python3 qwen14b_tp2_benchmark.py
"""

import asyncio
import aiohttp
import time
import json
import statistics
import sys

# ── Configuration ──────────────────────────────────────────────────
VLLM_URL = "http://localhost:8000/v1/completions"
MODEL = "Qwen/Qwen2.5-14B-Instruct"
PROMPT = "Explain the concept of GPU memory bandwidth and its impact on"
MAX_TOKENS = 50
TEMPERATURE = 0.0
WARMUP_REQUESTS = 5
NUM_ITERATIONS = 3

# Concurrency levels — adjusted for 14B model's lower capacity
# Max concurrency is ~15x at 4096 tokens, but we're generating 50 tokens
# so effective capacity is much higher for short sequences
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# ── Week 5 baselines (vLLM, Mistral 7B, single GPU) ──────────────
WEEK5_7B_BASELINES = {
    1: {"total": 53.2, "per_sample": 53.2},
}

# ── Week 4 baselines (vLLM, Llama 3.2 3B, single GPU) ────────────
WEEK4_3B_BASELINES = {
    1: {"total": 106.4, "per_sample": 106.4},
}

# ── Week 1 baselines (transformers, Llama 3.2 3B, single GPU) ────
WEEK1_BASELINES = {
    1: {"total": 84.0, "per_sample": 84.2},
    8: {"total": 609.0, "per_sample": 76.1},
    64: {"total": 3376.0, "per_sample": 52.7},
    128: {"total": 4291.0, "per_sample": 33.5},
    256: {"total": 4656.0, "per_sample": 18.2},
}


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
        async with session.post(
            VLLM_URL, json=payload, timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                return {
                    "success": False,
                    "error": f"HTTP {resp.status}: {error_text[:200]}",
                    "latency": time.perf_counter() - start,
                    "tokens": 0,
                }
            data = await resp.json()
            elapsed = time.perf_counter() - start
            tokens = data["usage"]["completion_tokens"]
            return {
                "success": True,
                "latency": elapsed,
                "tokens": tokens,
                "request_id": request_id,
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)[:200],
            "latency": time.perf_counter() - start,
            "tokens": 0,
        }


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

        iteration_results.append(
            {
                "wall_time": wall_time,
                "total_tokens": total_tokens,
                "successful": len(successful),
                "failed": len(failed),
                "total_throughput": total_throughput,
                "per_sample_throughput": per_sample_throughput,
                "latency_mean": statistics.mean(latencies),
                "latency_p50": statistics.median(latencies),
                "latency_p95": (
                    sorted(latencies)[int(0.95 * len(latencies))]
                    if len(latencies) > 1
                    else latencies[0]
                ),
                "latency_p99": (
                    sorted(latencies)[int(0.99 * len(latencies))]
                    if len(latencies) > 1
                    else latencies[0]
                ),
            }
        )

    return iteration_results


async def main():
    print("=" * 100)
    print("WEEK 6 EXPERIMENT 1: QWEN 2.5 14B (TP=2) THROUGHPUT BENCHMARK")
    print("=" * 100)
    print(f"Model:           {MODEL}")
    print(f"GPUs:            2x RTX 3090 (tensor parallel)")
    print(f"Generation:      {MAX_TOKENS} tokens per request")
    print(f"Warmup:          {WARMUP_REQUESTS} requests")
    print(f"Iterations:      {NUM_ITERATIONS} per concurrency level")
    print(f"Concurrency:     {CONCURRENCY_LEVELS}")
    print()

    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    async with aiohttp.ClientSession(connector=connector) as session:

        # ── Health check ──
        try:
            async with session.get(
                "http://localhost:8000/health", timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
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
                print(
                    f"  Warmup {i+1}: {result['tokens']} tokens in {result['latency']:.3f}s"
                )
        print()

        # ── Benchmark each concurrency level ──
        all_results = {}

        for conc in CONCURRENCY_LEVELS:
            print(f"── Concurrency {conc} ──")
            results = await benchmark_concurrency(session, conc, NUM_ITERATIONS)

            if not results:
                print(f"  All iterations failed at concurrency {conc}")
                print()
                continue

            avg_total_tp = statistics.mean(r["total_throughput"] for r in results)
            avg_per_sample = statistics.mean(
                r["per_sample_throughput"] for r in results
            )
            avg_latency = statistics.mean(r["latency_mean"] for r in results)
            avg_p95 = statistics.mean(r["latency_p95"] for r in results)
            avg_wall = statistics.mean(r["wall_time"] for r in results)

            all_results[conc] = {
                "total_throughput": avg_total_tp,
                "per_sample_throughput": avg_per_sample,
                "latency_mean": avg_latency,
                "latency_p95": avg_p95,
                "wall_time": avg_wall,
                "iterations": results,
            }

            print(
                f"  Total: {avg_total_tp:,.1f} tok/s | "
                f"Per-sample: {avg_per_sample:.1f} tok/s | "
                f"Latency mean: {avg_latency:.3f}s p95: {avg_p95:.3f}s"
            )
            print()

        # ── Results Summary ──
        print()
        print("=" * 120)
        print("RESULTS: QWEN 2.5 14B (TP=2, 2x RTX 3090)")
        print("=" * 120)
        print()
        print(
            f"{'Conc':>6} | {'14B Total':>10} {'14B/Sample':>11} | "
            f"{'Mean Lat':>9} {'P95 Lat':>9} | "
            f"{'vs W4 3B':>9} {'vs W1 TF':>9}"
        )
        print("-" * 120)

        for conc in CONCURRENCY_LEVELS:
            if conc not in all_results:
                print(f"{conc:>6} | {'FAILED':>10}")
                continue

            r = all_results[conc]

            # Find closest Week 4 baseline
            w4_concs = sorted(WEEK4_3B_BASELINES.keys())
            w4_match = min(w4_concs, key=lambda x: abs(x - conc)) if w4_concs else None
            w4_total = (
                WEEK4_3B_BASELINES.get(w4_match, {}).get("total", 0)
                if w4_match and abs(w4_match - conc) <= conc * 0.1
                else 0
            )

            # Find closest Week 1 baseline
            w1_concs = sorted(WEEK1_BASELINES.keys())
            w1_match = min(w1_concs, key=lambda x: abs(x - conc)) if w1_concs else None
            w1_total = (
                WEEK1_BASELINES.get(w1_match, {}).get("total", 0)
                if w1_match and abs(w1_match - conc) <= conc * 0.1
                else 0
            )

            w4_ratio = f"{r['total_throughput']/w4_total:.2f}x" if w4_total else "—"
            w1_ratio = f"{r['total_throughput']/w1_total:.2f}x" if w1_total else "—"

            print(
                f"{conc:>6} | "
                f"{r['total_throughput']:>8,.1f}   "
                f"{r['per_sample_throughput']:>9.1f}   | "
                f"{r['latency_mean']:>8.3f}s "
                f"{r['latency_p95']:>8.3f}s | "
                f"{w4_ratio:>9} "
                f"{w1_ratio:>9}"
            )

        # ── Key Metrics ──
        print()
        print("=" * 120)
        print("KEY METRICS")
        print("=" * 120)

        if 1 in all_results:
            r1 = all_results[1]
            print(
                f"  Single request:    {r1['total_throughput']:.1f} tok/s "
                f"(14B on 2 GPUs) vs 106 tok/s (3B on 1 GPU, Week 4)"
            )
            print(f"  Per-token latency: {1000/r1['total_throughput']:.1f} ms/tok")

        if all_results:
            peak_conc = max(
                all_results, key=lambda c: all_results[c]["total_throughput"]
            )
            peak_tp = all_results[peak_conc]["total_throughput"]
            print(
                f"  Peak throughput:   {peak_tp:,.1f} tok/s at concurrency={peak_conc}"
            )
            print(f"  vs Week 1 ceiling: {peak_tp/5000:.2f}x over transformers 3B peak")
            print(
                f"  vs Week 4 peak:    {peak_tp/6100:.2f}x over vLLM 3B single-GPU peak"
            )

        # Per-sample above 20 tok/s (usable threshold for 14B)
        above_20 = [
            c
            for c in CONCURRENCY_LEVELS
            if c in all_results and all_results[c]["per_sample_throughput"] >= 20
        ]
        if above_20:
            print(f"  Per-sample ≥20:    Up to concurrency={max(above_20)}")

        # KV cache analysis
        print()
        print("  KV Cache comparison:")
        print(f"    Llama 3.2 3B:   112 KB/token, 13.97 GiB pool, ~130K token capacity")
        print(f"    Qwen 2.5 14B:   192 KB/token, 5.81 GiB pool, 63,440 token capacity")
        print(f"    Memory ratio:    1.71x more per token, 2.05x less total capacity")
        print()


if __name__ == "__main__":
    asyncio.run(main())
