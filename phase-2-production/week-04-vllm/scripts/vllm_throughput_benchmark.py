#!/usr/bin/env python3
"""
Week 4 Experiment 2: vLLM Single-GPU Throughput Benchmark
Compare against Week 1 transformers baselines (Llama 3.2 3B, FP16)

Week 1 baselines (transformers, 50 tokens generated):
  Batch 1:    84 tok/s total,   84.2 tok/s per-sample
  Batch 8:   609 tok/s total,   76.1 tok/s per-sample
  Batch 64:  3,376 tok/s total, 52.7 tok/s per-sample
  Batch 128: 4,291 tok/s total, 33.5 tok/s per-sample
  Batch 256: 4,656 tok/s total, 18.2 tok/s per-sample
  Batch 512: 4,703 tok/s total,  9.2 tok/s per-sample
  Batch 1024: 4,951 tok/s total, 4.8 tok/s per-sample
  Batch 1200: 4,998 tok/s total, 4.2 tok/s per-sample

Usage:
  1. Start vLLM server:
     CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
         --model meta-llama/Llama-3.2-3B-Instruct --dtype float16 \
         --max-model-len 4096 --port 8000 --disable-log-requests

  2. Run benchmark:
     python3 scripts/vllm_throughput_benchmark.py
"""

import asyncio
import aiohttp
import time
import json
import statistics

# ── Configuration ──────────────────────────────────────────────────
VLLM_URL = "http://localhost:8000/v1/completions"
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
PROMPT = "Explain the concept of GPU memory bandwidth and its impact on"
MAX_TOKENS = 50          # Match Week 1 generation length
TEMPERATURE = 0.0        # Deterministic for reproducibility
WARMUP_REQUESTS = 5      # Warmup before each concurrency level
NUM_ITERATIONS = 3       # Repeat each level for stability

# Match Week 1 batch sizes
CONCURRENCY_LEVELS = [1, 8, 64, 128, 256, 512, 1024, 1200]

# Week 1 baselines for comparison
WEEK1_BASELINES = {
    1:    {"total": 84.0,   "per_sample": 84.2},
    8:    {"total": 609.0,  "per_sample": 76.1},
    64:   {"total": 3376.0, "per_sample": 52.7},
    128:  {"total": 4291.0, "per_sample": 33.5},
    256:  {"total": 4656.0, "per_sample": 18.2},
    512:  {"total": 4703.0, "per_sample": 9.2},
    1024: {"total": 4951.0, "per_sample": 4.8},
    1200: {"total": 4998.0, "per_sample": 4.2},
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
        async with session.post(VLLM_URL, json=payload) as resp:
            result = await resp.json()
            elapsed = time.perf_counter() - start
            
            if resp.status == 200:
                tokens = result["usage"]["completion_tokens"]
                return {
                    "request_id": request_id,
                    "tokens": tokens,
                    "latency": elapsed,
                    "success": True,
                }
            else:
                return {
                    "request_id": request_id,
                    "tokens": 0,
                    "latency": elapsed,
                    "success": False,
                    "error": result.get("error", str(resp.status)),
                }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            "request_id": request_id,
            "tokens": 0,
            "latency": elapsed,
            "success": False,
            "error": str(e),
        }


async def run_concurrent_batch(session, num_requests):
    """Send num_requests concurrently and collect all results."""
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
            "latency_p95": sorted(latencies)[int(0.95 * len(latencies))] if len(latencies) > 1 else latencies[0],
            "latency_p99": sorted(latencies)[int(0.99 * len(latencies))] if len(latencies) > 1 else latencies[0],
        })
    
    return iteration_results


async def main():
    print("=" * 80)
    print("WEEK 4 EXPERIMENT 2: vLLM SINGLE-GPU THROUGHPUT BENCHMARK")
    print("=" * 80)
    print(f"Model:           {MODEL}")
    print(f"Generation:      {MAX_TOKENS} tokens per request")
    print(f"Prompt:          \"{PROMPT[:50]}...\"")
    print(f"Iterations:      {NUM_ITERATIONS} per concurrency level")
    print(f"Concurrency:     {CONCURRENCY_LEVELS}")
    print()
    
    # Use a large connection pool and long timeouts for high concurrency
    timeout = aiohttp.ClientTimeout(total=300)
    connector = aiohttp.TCPConnector(limit=0)  # No connection limit
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # ── Warmup ──
        print(f"Warming up with {WARMUP_REQUESTS} requests...")
        await run_concurrent_batch(session, WARMUP_REQUESTS)
        print("Warmup complete.\n")
        
        # ── Benchmark each concurrency level ──
        all_results = {}
        
        for conc in CONCURRENCY_LEVELS:
            print(f"Testing concurrency={conc}...")
            
            results = await benchmark_concurrency(session, conc, NUM_ITERATIONS)
            
            if not results:
                print(f"  FAILED - no successful iterations\n")
                continue
            
            # Average across iterations
            avg_total_tp = statistics.mean(r["total_throughput"] for r in results)
            avg_per_sample = statistics.mean(r["per_sample_throughput"] for r in results)
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
            
            print(f"  Total: {avg_total_tp:,.1f} tok/s | "
                  f"Per-sample: {avg_per_sample:.1f} tok/s | "
                  f"Latency mean: {avg_latency:.3f}s p95: {avg_p95:.3f}s")
            print()
        
        # ── Results Summary ──
        print()
        print("=" * 100)
        print("RESULTS SUMMARY: vLLM vs TRANSFORMERS (Week 1)")
        print("=" * 100)
        print()
        print(f"{'Conc':>6} | {'vLLM Total':>12} {'TF Total':>12} {'Speedup':>8} | "
              f"{'vLLM/Sample':>12} {'TF/Sample':>12} {'Speedup':>8} | "
              f"{'Mean Lat':>9} {'P95 Lat':>9}")
        print("-" * 100)
        
        for conc in CONCURRENCY_LEVELS:
            if conc not in all_results:
                print(f"{conc:>6} | {'FAILED':>12}")
                continue
            
            r = all_results[conc]
            w1 = WEEK1_BASELINES.get(conc, {})
            
            w1_total = w1.get("total", 0)
            w1_per = w1.get("per_sample", 0)
            
            total_speedup = r["total_throughput"] / w1_total if w1_total else 0
            per_speedup = r["per_sample_throughput"] / w1_per if w1_per else 0
            
            print(f"{conc:>6} | "
                  f"{r['total_throughput']:>10,.1f}  "
                  f"{w1_total:>10,.1f}  "
                  f"{total_speedup:>6.2f}x  | "
                  f"{r['per_sample_throughput']:>10.1f}  "
                  f"{w1_per:>10.1f}  "
                  f"{per_speedup:>6.2f}x  | "
                  f"{r['latency_mean']:>8.3f}s "
                  f"{r['latency_p95']:>8.3f}s")
        
        print()
        print("=" * 100)
        print("KEY METRICS")
        print("=" * 100)
        
        if 1 in all_results:
            r1 = all_results[1]
            print(f"  Single request:    {r1['total_throughput']:.1f} tok/s "
                  f"(Week 1: {WEEK1_BASELINES[1]['total']:.1f} tok/s, "
                  f"{r1['total_throughput']/WEEK1_BASELINES[1]['total']:.2f}x)")
        
        # Find peak total throughput
        if all_results:
            peak_conc = max(all_results, key=lambda c: all_results[c]["total_throughput"])
            peak_tp = all_results[peak_conc]["total_throughput"]
            print(f"  Peak throughput:   {peak_tp:,.1f} tok/s at concurrency={peak_conc} "
                  f"(Week 1 peak: ~5,000 tok/s)")
            print(f"  Peak improvement:  {peak_tp/5000:.2f}x over transformers ceiling")
        
        # Find where per-sample stays above 50 tok/s
        above_50 = [c for c in CONCURRENCY_LEVELS 
                     if c in all_results and all_results[c]["per_sample_throughput"] >= 50]
        if above_50:
            print(f"  Per-sample ≥50:    Up to concurrency={max(above_50)}")
        
        print()


if __name__ == "__main__":
    asyncio.run(main())
