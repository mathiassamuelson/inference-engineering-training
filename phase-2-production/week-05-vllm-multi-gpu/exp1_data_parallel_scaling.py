"""
Week 5 - Experiment 1: vLLM Data Parallel Scaling (4x RTX 3090)
Launches independent vLLM instances on each GPU and measures
total system throughput under concurrent load.

Comparison targets:
  - Week 3 manual data parallel: 7,422 tok/s (4 GPU, batch=32)
  - Week 4 vLLM single-GPU:     ~6,100 tok/s peak, 106 tok/s single-request
"""

import subprocess
import time
import asyncio
import aiohttp
import json
import signal
import sys
from dataclasses import dataclass, field

# ── Configuration ──────────────────────────────────────────────
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
GPU_IDS = [0, 1, 2, 3]
BASE_PORT = 8000  # ports 8000, 8001, 8002, 8003
MAX_MODEL_LEN = 4096

# Test parameters
PROMPT = "Explain the concept of continuous batching in LLM inference and why it matters for production deployments."
OUTPUT_LEN = 128
WARMUP_REQUESTS = 5
CONCURRENCY_LEVELS = [1, 4, 8, 16, 32, 64]  # per GPU


@dataclass
class ServerProcess:
    gpu_id: int
    port: int
    process: subprocess.Popen = None


@dataclass
class RequestResult:
    gpu_id: int
    tokens_generated: int
    latency_s: float
    tokens_per_sec: float


def launch_vllm_server(gpu_id: int, port: int) -> subprocess.Popen:
    """Launch a vLLM OpenAI-compatible server on a specific GPU."""
    env = {
        **dict(__import__('os').environ),
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
    }
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--port", str(port),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--dtype", "float16",
        "--disable-log-requests",
        "--gpu-memory-utilization", "0.90",
    ]
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc


async def wait_for_server(port: int, timeout: int = 300):
    """Poll the health endpoint until the server is ready."""
    url = f"http://localhost:{port}/health"
    start = time.time()
    async with aiohttp.ClientSession() as session:
        while time.time() - start < timeout:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(2)
    raise TimeoutError(f"Server on port {port} did not start within {timeout}s")


async def send_request(session: aiohttp.ClientSession, port: int,
                       gpu_id: int) -> RequestResult:
    """Send a single completion request and measure throughput."""
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": OUTPUT_LEN,
        "temperature": 0.0,
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
        tokens_per_sec=tokens / elapsed,
    )


async def benchmark_gpu(port: int, gpu_id: int, concurrency: int,
                        num_requests: int) -> list[RequestResult]:
    """Send num_requests to a single GPU server at given concurrency."""
    results = []
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    timeout = aiohttp.ClientTimeout(total=120)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async def bounded_request():
            async with semaphore:
                return await send_request(session, port, gpu_id)

        tasks = [bounded_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)

    return list(results)


async def benchmark_all_gpus(servers: list[ServerProcess], concurrency_per_gpu: int,
                              requests_per_gpu: int) -> list[RequestResult]:
    """Run benchmarks across all GPU servers simultaneously."""
    tasks = [
        benchmark_gpu(s.port, s.gpu_id, concurrency_per_gpu, requests_per_gpu)
        for s in servers
    ]
    all_results = await asyncio.gather(*tasks)
    # Flatten
    return [r for gpu_results in all_results for r in gpu_results]


def print_results(results: list[RequestResult], concurrency_per_gpu: int,
                  num_gpus: int, wall_time: float):
    """Print formatted benchmark results."""
    total_tokens = sum(r.tokens_generated for r in results)
    total_requests = len(results)
    avg_latency = sum(r.latency_s for r in results) / total_requests
    system_throughput = total_tokens / wall_time

    # Per-GPU breakdown
    per_gpu = {}
    for r in results:
        per_gpu.setdefault(r.gpu_id, []).append(r)

    print(f"\n{'='*70}")
    print(f"  Concurrency: {concurrency_per_gpu}/GPU × {num_gpus} GPUs "
          f"= {concurrency_per_gpu * num_gpus} total")
    print(f"{'='*70}")
    print(f"  System throughput:  {system_throughput:,.1f} tok/s")
    print(f"  Total requests:     {total_requests}")
    print(f"  Total tokens:       {total_tokens:,}")
    print(f"  Wall clock time:    {wall_time:.2f}s")
    print(f"  Avg latency:        {avg_latency:.3f}s")
    print(f"  Per-request avg:    {sum(r.tokens_per_sec for r in results)/total_requests:.1f} tok/s")
    print()
    print(f"  {'GPU':>4}  {'Requests':>9}  {'Tokens':>8}  {'Avg Latency':>12}  {'Avg tok/s':>10}")
    print(f"  {'---':>4}  {'--------':>9}  {'------':>8}  {'-----------':>12}  {'---------':>10}")
    for gpu_id in sorted(per_gpu.keys()):
        gr = per_gpu[gpu_id]
        g_tokens = sum(r.tokens_generated for r in gr)
        g_lat = sum(r.latency_s for r in gr) / len(gr)
        g_tps = sum(r.tokens_per_sec for r in gr) / len(gr)
        print(f"  {gpu_id:>4}  {len(gr):>9}  {g_tokens:>8,}  {g_lat:>11.3f}s  {g_tps:>9.1f}")


def cleanup_servers(servers: list[ServerProcess]):
    """Terminate all vLLM server processes."""
    for s in servers:
        if s.process and s.process.poll() is None:
            s.process.terminate()
            try:
                s.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                s.process.kill()


async def main():
    servers = []

    # ── Phase 1: Launch servers ────────────────────────────────
    print(f"\n{'#'*70}")
    print(f"  Week 5 Experiment 1: vLLM Data Parallel Scaling")
    print(f"  Model: {MODEL}")
    print(f"  GPUs: {len(GPU_IDS)} × RTX 3090")
    print(f"{'#'*70}\n")

    # First, test with subset of GPUs to measure scaling
    for num_gpus in [1, 2, 4]:
        gpu_subset = GPU_IDS[:num_gpus]
        servers = []

        print(f"\n{'='*70}")
        print(f"  Launching {num_gpus} vLLM server(s)...")
        print(f"{'='*70}")

        for gpu_id in gpu_subset:
            port = BASE_PORT + gpu_id
            proc = launch_vllm_server(gpu_id, port)
            servers.append(ServerProcess(gpu_id=gpu_id, port=port, process=proc))
            print(f"  GPU {gpu_id} → port {port} (pid {proc.pid})")

        # Wait for all servers to be ready
        print(f"\n  Waiting for servers to load model...")
        try:
            await asyncio.gather(*[
                wait_for_server(s.port) for s in servers
            ])
        except TimeoutError as e:
            print(f"  ERROR: {e}")
            cleanup_servers(servers)
            continue

        print(f"  All {num_gpus} server(s) ready.\n")

        # ── Phase 2: Warmup ────────────────────────────────────
        print(f"  Warming up ({WARMUP_REQUESTS} requests per GPU)...")
        await benchmark_all_gpus(servers, concurrency_per_gpu=1,
                                  requests_per_gpu=WARMUP_REQUESTS)

        # ── Phase 3: Benchmark at each concurrency level ──────
        for conc in CONCURRENCY_LEVELS:
            requests_per_gpu = max(conc * 3, 30)  # enough to get stable measurement

            wall_start = time.time()
            results = await benchmark_all_gpus(servers, conc, requests_per_gpu)
            wall_time = time.time() - wall_start

            print_results(results, conc, num_gpus, wall_time)

        # Cleanup before next GPU count
        print(f"\n  Shutting down {num_gpus} server(s)...")
        cleanup_servers(servers)
        await asyncio.sleep(5)  # let GPU memory fully release

    # ── Summary comparison ─────────────────────────────────────
    print(f"\n{'#'*70}")
    print(f"  Comparison Targets:")
    print(f"    Week 3 manual data parallel (4 GPU): 7,422 tok/s")
    print(f"    Week 4 vLLM single-GPU peak:         ~6,100 tok/s")
    print(f"    Week 4 vLLM single-request:          106 tok/s")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted — cleaning up...")
        sys.exit(1)
