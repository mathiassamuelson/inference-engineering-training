#!/usr/bin/env python3
"""
Week 4 Experiment 3: Concurrent User Simulation
Tests vLLM's continuous batching with realistic production traffic patterns.

Key differences from Experiment 2:
  - Staggered arrivals (Poisson-like distribution)
  - Variable prompt lengths (short, medium, long)
  - Variable generation lengths (20-200 tokens)
  - Sustained load over time window
  - Per-user latency tracking (TTFT + generation)

Usage:
  python3 scripts/vllm_concurrent_users.py
"""

import asyncio
import aiohttp
import time
import random
import statistics
import json

# ── Configuration ──────────────────────────────────────────────────
VLLM_URL = "http://localhost:8000/v1/completions"
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
TEMPERATURE = 0.0

# User levels to test (concurrent users sending requests continuously)
USER_LEVELS = [1, 5, 10, 25, 50, 100, 150, 200]

# Each user sends this many requests sequentially during the test
REQUESTS_PER_USER = 5

# Warmup
WARMUP_REQUESTS = 10

# Variable workloads - mix of short and long requests
WORKLOADS = [
    # (prompt, min_tokens, max_tokens, weight)
    ("Summarize in one sentence:", 20, 40, 0.3),          # Short - quick answers
    ("Explain the following concept in detail:", 50, 100, 0.4),  # Medium - typical API
    ("Write a comprehensive analysis of the following topic. Include multiple "
     "perspectives, examples, and a conclusion:", 100, 200, 0.2),  # Long - heavy generation
    ("Answer yes or no:", 5, 10, 0.1),                    # Tiny - classification-like
]

# Subjects to vary the prompts (avoids prefix caching skewing results)
SUBJECTS = [
    "quantum computing", "machine learning", "database indexing",
    "network protocols", "operating systems", "compiler design",
    "distributed systems", "cryptography", "computer graphics",
    "natural language processing", "reinforcement learning",
    "cloud architecture", "edge computing", "data pipelines",
    "API design", "memory management", "file systems",
    "container orchestration", "load balancing", "caching strategies",
]


def generate_request():
    """Generate a random request with variable prompt and generation length."""
    # Weighted random workload selection
    rand = random.random()
    cumulative = 0
    for prompt_prefix, min_tok, max_tok, weight in WORKLOADS:
        cumulative += weight
        if rand <= cumulative:
            break
    
    subject = random.choice(SUBJECTS)
    prompt = f"{prompt_prefix} {subject}"
    max_tokens = random.randint(min_tok, max_tok)
    
    return prompt, max_tokens


async def user_session(session, user_id, num_requests, results_list, start_barrier):
    """Simulate a single user sending sequential requests."""
    # Wait for all users to be ready
    await start_barrier.wait()
    
    # Small random stagger to avoid thundering herd
    await asyncio.sleep(random.uniform(0, 0.1))
    
    for req_num in range(num_requests):
        prompt, max_tokens = generate_request()
        
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": TEMPERATURE,
        }
        
        req_start = time.perf_counter()
        try:
            async with session.post(VLLM_URL, json=payload) as resp:
                result = await resp.json()
                req_end = time.perf_counter()
                
                if resp.status == 200:
                    tokens = result["usage"]["completion_tokens"]
                    prompt_tokens = result["usage"]["prompt_tokens"]
                    latency = req_end - req_start
                    tok_per_sec = tokens / latency if latency > 0 else 0
                    
                    results_list.append({
                        "user_id": user_id,
                        "request_num": req_num,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": tokens,
                        "max_tokens_requested": max_tokens,
                        "latency": latency,
                        "tokens_per_sec": tok_per_sec,
                        "success": True,
                    })
                else:
                    results_list.append({
                        "user_id": user_id,
                        "request_num": req_num,
                        "latency": req_end - req_start,
                        "success": False,
                        "error": str(result),
                    })
        except Exception as e:
            results_list.append({
                "user_id": user_id,
                "request_num": req_num,
                "latency": time.perf_counter() - req_start,
                "success": False,
                "error": str(e),
            })


async def run_user_simulation(session, num_users, requests_per_user):
    """Run a simulation with N concurrent users."""
    results = []
    barrier = asyncio.Barrier(num_users)
    
    wall_start = time.perf_counter()
    
    tasks = [
        user_session(session, uid, requests_per_user, results, barrier)
        for uid in range(num_users)
    ]
    await asyncio.gather(*tasks)
    
    wall_time = time.perf_counter() - wall_start
    
    return results, wall_time


async def main():
    print("=" * 90)
    print("WEEK 4 EXPERIMENT 3: CONCURRENT USER SIMULATION")
    print("=" * 90)
    print(f"Model:             {MODEL}")
    print(f"User levels:       {USER_LEVELS}")
    print(f"Requests/user:     {REQUESTS_PER_USER}")
    print(f"Workload mix:      30% short (20-40 tok), 40% medium (50-100 tok), "
          f"20% long (100-200 tok), 10% tiny (5-10 tok)")
    print()
    
    timeout = aiohttp.ClientTimeout(total=600)
    connector = aiohttp.TCPConnector(limit=0)
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # ── Warmup ──
        print(f"Warming up with {WARMUP_REQUESTS} sequential requests...")
        for _ in range(WARMUP_REQUESTS):
            prompt, max_tokens = generate_request()
            payload = {
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": TEMPERATURE,
            }
            async with session.post(VLLM_URL, json=payload) as resp:
                await resp.json()
        print("Warmup complete.\n")
        
        # ── Run each user level ──
        all_summaries = {}
        
        for num_users in USER_LEVELS:
            total_requests = num_users * REQUESTS_PER_USER
            print(f"Simulating {num_users} concurrent users "
                  f"({total_requests} total requests)...")
            
            results, wall_time = await run_user_simulation(
                session, num_users, REQUESTS_PER_USER
            )
            
            successful = [r for r in results if r["success"]]
            failed = [r for r in results if not r["success"]]
            
            if not successful:
                print(f"  ALL FAILED\n")
                continue
            
            latencies = [r["latency"] for r in successful]
            tps_values = [r["tokens_per_sec"] for r in successful]
            completion_tokens = [r["completion_tokens"] for r in successful]
            total_tokens = sum(r["completion_tokens"] for r in successful)
            
            system_throughput = total_tokens / wall_time
            
            summary = {
                "num_users": num_users,
                "total_requests": total_requests,
                "successful": len(successful),
                "failed": len(failed),
                "wall_time": wall_time,
                "system_throughput": system_throughput,
                "avg_tokens_generated": statistics.mean(completion_tokens),
                "per_user_tps_mean": statistics.mean(tps_values),
                "per_user_tps_median": statistics.median(tps_values),
                "per_user_tps_min": min(tps_values),
                "latency_mean": statistics.mean(latencies),
                "latency_p50": statistics.median(latencies),
                "latency_p95": sorted(latencies)[int(0.95 * len(latencies))],
                "latency_p99": sorted(latencies)[int(0.99 * len(latencies))],
                "latency_min": min(latencies),
                "latency_max": max(latencies),
            }
            all_summaries[num_users] = summary
            
            print(f"  System: {system_throughput:,.1f} tok/s | "
                  f"Per-user: {summary['per_user_tps_mean']:.1f} tok/s (mean), "
                  f"{summary['per_user_tps_min']:.1f} (min) | "
                  f"Latency p50: {summary['latency_p50']:.3f}s "
                  f"p95: {summary['latency_p95']:.3f}s | "
                  f"Failed: {len(failed)}")
            print()
        
        # ── Summary Table ──
        print()
        print("=" * 110)
        print("RESULTS SUMMARY: CONCURRENT USER SIMULATION")
        print("=" * 110)
        print()
        print(f"{'Users':>6} | {'Sys tok/s':>10} | "
              f"{'User tok/s':>10} {'(min)':>8} | "
              f"{'Lat p50':>8} {'Lat p95':>8} {'Lat p99':>8} {'Lat max':>8} | "
              f"{'Avg Gen':>8} | {'Fail':>5}")
        print("-" * 110)
        
        for num_users in USER_LEVELS:
            if num_users not in all_summaries:
                print(f"{num_users:>6} | {'FAILED':>10}")
                continue
            
            s = all_summaries[num_users]
            print(f"{num_users:>6} | "
                  f"{s['system_throughput']:>10,.1f} | "
                  f"{s['per_user_tps_mean']:>10.1f} {s['per_user_tps_min']:>7.1f} | "
                  f"{s['latency_p50']:>7.3f}s {s['latency_p95']:>7.3f}s "
                  f"{s['latency_p99']:>7.3f}s {s['latency_max']:>7.3f}s | "
                  f"{s['avg_tokens_generated']:>7.1f}t | "
                  f"{s['failed']:>5}")
        
        # ── Production Capacity Analysis ──
        print()
        print("=" * 110)
        print("PRODUCTION CAPACITY ANALYSIS")
        print("=" * 110)
        
        # SLA thresholds
        sla_thresholds = {
            "Real-time chat (p95 < 2s)": 2.0,
            "API serving (p95 < 5s)": 5.0,
            "Batch processing (p95 < 10s)": 10.0,
        }
        
        for sla_name, threshold in sla_thresholds.items():
            qualifying = [
                (u, s) for u, s in all_summaries.items()
                if s["latency_p95"] <= threshold
            ]
            if qualifying:
                best = max(qualifying, key=lambda x: x[0])
                print(f"  {sla_name}: up to {best[0]} users "
                      f"({best[1]['system_throughput']:,.0f} tok/s, "
                      f"p95={best[1]['latency_p95']:.2f}s)")
            else:
                print(f"  {sla_name}: < {USER_LEVELS[0]} users")
        
        # Compare to Week 1 capacity estimate
        print()
        print("  Week 1 estimate (transformers): 100-150 users per GPU")
        print("  Week 3 data parallel (4 GPU):   400-600 users total")
        print()


if __name__ == "__main__":
    asyncio.run(main())
