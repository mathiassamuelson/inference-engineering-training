"""
Week 5 - Experiment 3: Sustained Load Testing (4x RTX 3090)
Simulates continuous production traffic with mixed request lengths
across 4 vLLM data-parallel instances.

Key question: Does throughput remain stable under sustained load,
and how do mixed request lengths affect latency distribution?
"""

import subprocess
import time
import asyncio
import aiohttp
import random
import os
import sys
import statistics
from dataclasses import dataclass, field

# ── Configuration ──────────────────────────────────────────────
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
GPU_IDS = [0, 1, 2, 3]
BASE_PORT = 8000
MAX_MODEL_LEN = 4096

# Sustained load parameters
TEST_DURATION_S = 60  # each test phase runs for 60 seconds
CONCURRENCY = 32  # 32 requests in flight across all GPUs
RAMP_UP_S = 5  # gradual ramp-up period

# Request profiles
UNIFORM_OUTPUT_LEN = 128

# Mixed workload: simulates real chat traffic
MIXED_PROFILES = [
    {"name": "quick_reply", "weight": 0.40, "max_tokens": 32},
    {"name": "short_answer", "weight": 0.30, "max_tokens": 128},
    {"name": "explanation", "weight": 0.20, "max_tokens": 256},
    {"name": "long_response", "weight": 0.10, "max_tokens": 512},
]

PROMPTS = [
    "What is the capital of France?",
    "Explain how a transformer model processes a sentence step by step.",
    "Write a brief comparison of TCP and UDP protocols.",
    "What are three advantages of using containerized deployments?",
    "Describe the PagedAttention mechanism used in vLLM.",
    "Summarize the key differences between data parallelism and tensor parallelism for GPU inference.",
    "What is continuous batching and why does it matter?",
    "Explain the concept of KV cache in autoregressive language models.",
]


@dataclass
class RequestResult:
    gpu_id: int
    tokens_generated: int
    latency_s: float
    output_len_requested: int
    profile_name: str
    timestamp: float  # when the request completed


def pick_mixed_profile() -> dict:
    """Weighted random selection of request profile."""
    r = random.random()
    cumulative = 0.0
    for profile in MIXED_PROFILES:
        cumulative += profile["weight"]
        if r <= cumulative:
            return profile
    return MIXED_PROFILES[-1]


def launch_vllm_server(gpu_id: int, port: int) -> subprocess.Popen:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL,
        "--port",
        str(port),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--dtype",
        "float16",
        "--disable-log-requests",
        "--gpu-memory-utilization",
        "0.90",
    ]
    return subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


async def wait_for_server(port: int, timeout: int = 300):
    url = f"http://localhost:{port}/health"
    start = time.time()
    async with aiohttp.ClientSession() as session:
        while time.time() - start < timeout:
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        return
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(2)
    raise TimeoutError(f"Server on port {port} not ready within {timeout}s")


async def send_request(
    session: aiohttp.ClientSession,
    port: int,
    gpu_id: int,
    max_tokens: int,
    profile_name: str,
) -> RequestResult:
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": MODEL,
        "prompt": random.choice(PROMPTS),
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    start = time.time()
    async with session.post(url, json=payload) as resp:
        data = await resp.json()
        elapsed = time.time() - start
    tokens = data["usage"]["completion_tokens"]
    return RequestResult(
        gpu_id=gpu_id,
        tokens_generated=tokens,
        latency_s=elapsed,
        output_len_requested=max_tokens,
        profile_name=profile_name,
        timestamp=time.time(),
    )


async def sustained_load(
    ports: list[int],
    gpu_ids: list[int],
    duration_s: float,
    concurrency: int,
    mixed: bool,
) -> list[RequestResult]:
    """
    Maintain `concurrency` in-flight requests across all servers
    for `duration_s` seconds. As each request completes, immediately
    launch a replacement — this is the continuous traffic pattern.
    Requests still in-flight at deadline are cancelled (not drained).
    """
    results = []
    end_time = time.time() + duration_s
    num_gpus = len(ports)

    connector = aiohttp.TCPConnector(limit=concurrency + 20)
    timeout = aiohttp.ClientTimeout(total=120)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        active_tasks: set[asyncio.Task] = set()
        request_counter = 0

        def make_request():
            nonlocal request_counter
            idx = request_counter % num_gpus
            request_counter += 1
            port = ports[idx]
            gpu_id = gpu_ids[idx]

            if mixed:
                profile = pick_mixed_profile()
                max_tokens = profile["max_tokens"]
                name = profile["name"]
            else:
                max_tokens = UNIFORM_OUTPUT_LEN
                name = "uniform"

            return asyncio.create_task(
                send_request(session, port, gpu_id, max_tokens, name)
            )

        # Ramp up
        for _ in range(concurrency):
            if time.time() >= end_time:
                break
            task = make_request()
            active_tasks.add(task)

        # Sustained loop: replace completed requests immediately
        while time.time() < end_time and active_tasks:
            remaining = end_time - time.time()
            if remaining <= 0:
                break

            done, active_tasks = await asyncio.wait(
                active_tasks,
                return_when=asyncio.FIRST_COMPLETED,
                timeout=remaining,
            )

            for task in done:
                try:
                    result = task.result()
                    results.append(result)
                except Exception as e:
                    print(f"    Request error: {e}")

                # Replace with a new request if still within duration
                if time.time() < end_time:
                    new_task = make_request()
                    active_tasks.add(new_task)

        # Cancel any remaining in-flight tasks instead of draining
        for task in active_tasks:
            task.cancel()

        # Wait briefly for cancellations to propagate
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)

    return results


def analyze_results(results: list[RequestResult], phase_name: str, wall_time: float):
    """Print detailed analysis of sustained load results."""
    if not results:
        print(f"  No results for {phase_name}")
        return

    latencies = [r.latency_s for r in results]
    tok_per_s = [r.tokens_generated / r.latency_s for r in results]
    total_tokens = sum(r.tokens_generated for r in results)

    print(f"\n  {'='*66}")
    print(f"  {phase_name}")
    print(f"  {'='*66}")
    print(f"  Duration:           {wall_time:.1f}s")
    print(f"  Total requests:     {len(results)}")
    print(f"  Total tokens:       {total_tokens:,}")
    print(f"  System throughput:  {total_tokens / wall_time:,.1f} tok/s")
    print(f"  Request rate:       {len(results) / wall_time:.1f} req/s")
    print()
    print(f"  Latency distribution:")
    print(f"    p50:  {statistics.median(latencies):.3f}s")
    print(f"    p95:  {sorted(latencies)[int(len(latencies)*0.95)]:.3f}s")
    print(f"    p99:  {sorted(latencies)[int(len(latencies)*0.99)]:.3f}s")
    print(f"    min:  {min(latencies):.3f}s")
    print(f"    max:  {max(latencies):.3f}s")
    print()
    print(f"  Per-request throughput:")
    print(f"    avg:  {statistics.mean(tok_per_s):.1f} tok/s")
    print(f"    min:  {min(tok_per_s):.1f} tok/s")
    print(f"    max:  {max(tok_per_s):.1f} tok/s")

    # Per-GPU breakdown
    per_gpu = {}
    for r in results:
        per_gpu.setdefault(r.gpu_id, []).append(r)

    print()
    print(
        f"  {'GPU':>5}  {'Reqs':>6}  {'Tokens':>8}  {'Avg Lat':>9}  {'p95 Lat':>9}  {'tok/s':>7}"
    )
    print(
        f"  {'---':>5}  {'----':>6}  {'------':>8}  {'-------':>9}  {'-------':>9}  {'-----':>7}"
    )
    for gid in sorted(per_gpu.keys()):
        gr = per_gpu[gid]
        gl = sorted([r.latency_s for r in gr])
        gt = sum(r.tokens_generated for r in gr)
        print(
            f"  {gid:>5}  {len(gr):>6}  {gt:>8,}  "
            f"{statistics.mean(gl):>8.3f}s  "
            f"{gl[int(len(gl)*0.95)]:>8.3f}s  "
            f"{sum(r.tokens_generated/r.latency_s for r in gr)/len(gr):>6.1f}"
        )

    # Per-profile breakdown (mixed workload only)
    profiles = {}
    for r in results:
        profiles.setdefault(r.profile_name, []).append(r)

    if len(profiles) > 1:
        print()
        print(f"  Request profile breakdown:")
        print(
            f"  {'Profile':<16}  {'Count':>6}  {'Avg Tokens':>11}  "
            f"{'Avg Lat':>9}  {'p95 Lat':>9}  {'tok/s':>7}"
        )
        print(
            f"  {'-'*15:<16}  {'-----':>6}  {'----------':>11}  "
            f"{'-------':>9}  {'-------':>9}  {'-----':>7}"
        )
        for name in ["quick_reply", "short_answer", "explanation", "long_response"]:
            if name not in profiles:
                continue
            pr = profiles[name]
            pl = sorted([r.latency_s for r in pr])
            avg_tok = sum(r.tokens_generated for r in pr) / len(pr)
            avg_tps = sum(r.tokens_generated / r.latency_s for r in pr) / len(pr)
            print(
                f"  {name:<16}  {len(pr):>6}  {avg_tok:>10.1f}  "
                f"{statistics.mean(pl):>8.3f}s  "
                f"{pl[int(len(pl)*0.95)]:>8.3f}s  "
                f"{avg_tps:>6.1f}"
            )


def check_stability(
    results: list[RequestResult], wall_time: float, window_s: float = 10.0
):
    """Check throughput stability across time windows."""
    if not results:
        return

    min_ts = min(r.timestamp for r in results)
    max_ts = max(r.timestamp for r in results)
    duration = max_ts - min_ts

    num_windows = int(duration / window_s)
    if num_windows < 2:
        return

    print(f"\n  Throughput stability ({window_s:.0f}s windows):")
    print(f"  {'Window':>8}  {'Requests':>9}  {'Tokens':>8}  {'tok/s':>8}")
    print(f"  {'------':>8}  {'--------':>9}  {'------':>8}  {'-----':>8}")

    window_throughputs = []
    for i in range(num_windows):
        w_start = min_ts + i * window_s
        w_end = w_start + window_s
        w_results = [r for r in results if w_start <= r.timestamp < w_end]
        w_tokens = sum(r.tokens_generated for r in w_results)
        w_tps = w_tokens / window_s
        window_throughputs.append(w_tps)
        print(
            f"  {i*window_s:>6.0f}-{(i+1)*window_s:>2.0f}s  "
            f"{len(w_results):>9}  {w_tokens:>8,}  {w_tps:>7,.1f}"
        )

    if len(window_throughputs) >= 2:
        cv = statistics.stdev(window_throughputs) / statistics.mean(window_throughputs)
        print(
            f"\n  Coefficient of variation: {cv:.3f} "
            f"({'stable' if cv < 0.1 else 'unstable'})"
        )


async def main():
    print(f"\n{'#'*70}")
    print(f"  Week 5 Experiment 3: Sustained Load Test")
    print(f"  Model: {MODEL}")
    print(f"  GPUs: {len(GPU_IDS)} × RTX 3090")
    print(f"  Concurrency: {CONCURRENCY} total")
    print(f"  Duration: {TEST_DURATION_S}s per phase")
    print(f"{'#'*70}\n")

    # Launch all 4 servers
    servers = []
    ports = []
    for gpu_id in GPU_IDS:
        port = BASE_PORT + gpu_id
        proc = launch_vllm_server(gpu_id, port)
        servers.append(proc)
        ports.append(port)
        print(f"  GPU {gpu_id} → port {port} (pid {proc.pid})")

    try:
        print(f"\n  Waiting for all servers to load model...")
        await asyncio.gather(*[wait_for_server(p) for p in ports])
        print(f"  All servers ready.\n")

        # Warmup
        print(f"  Warming up...")
        connector = aiohttp.TCPConnector(limit=20)
        async with aiohttp.ClientSession(connector=connector) as session:
            warmup_tasks = []
            for port, gpu_id in zip(ports, GPU_IDS):
                for _ in range(3):
                    warmup_tasks.append(
                        send_request(session, port, gpu_id, 64, "warmup")
                    )
            await asyncio.gather(*warmup_tasks)
        print(f"  Warmup complete.\n")

        # ── Phase A: Uniform requests ──────────────────────────
        print(f"  Phase A: Uniform requests (all {UNIFORM_OUTPUT_LEN} tokens)")
        print(f"  Running for {TEST_DURATION_S}s...")
        wall_start = time.time()
        uniform_results = await sustained_load(
            ports, GPU_IDS, TEST_DURATION_S, CONCURRENCY, mixed=False
        )
        uniform_wall = time.time() - wall_start
        analyze_results(uniform_results, "Phase A: Uniform Requests", uniform_wall)
        check_stability(uniform_results, uniform_wall)

        await asyncio.sleep(3)  # brief cooldown

        # ── Phase B: Mixed requests ────────────────────────────
        print(f"\n\n  Phase B: Mixed requests (32-512 tokens, weighted)")
        print(f"  Running for {TEST_DURATION_S}s...")
        wall_start = time.time()
        mixed_results = await sustained_load(
            ports, GPU_IDS, TEST_DURATION_S, CONCURRENCY, mixed=True
        )
        mixed_wall = time.time() - wall_start
        analyze_results(mixed_results, "Phase B: Mixed Requests", mixed_wall)
        check_stability(mixed_results, mixed_wall)

        # ── Comparison ─────────────────────────────────────────
        print(f"\n\n{'#'*70}")
        print(f"  Phase A vs Phase B Summary:")
        if uniform_results and mixed_results:
            u_tps = sum(r.tokens_generated for r in uniform_results) / uniform_wall
            m_tps = sum(r.tokens_generated for r in mixed_results) / mixed_wall
            u_rps = len(uniform_results) / uniform_wall
            m_rps = len(mixed_results) / mixed_wall
            print(f"    Uniform: {u_tps:,.1f} tok/s, {u_rps:.1f} req/s")
            print(f"    Mixed:   {m_tps:,.1f} tok/s, {m_rps:.1f} req/s")
            print(f"    Mixed request rate advantage: {m_rps/u_rps:.2f}x")
        print(f"{'#'*70}\n")

    finally:
        # Kill entire process groups to catch vLLM child processes
        import signal

        for proc in servers:
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
        time.sleep(3)
        for proc in servers:
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
