"""
Week 5 - Experiment 2: Larger Model Benchmark (Mistral 7B)
Single-GPU vLLM serving to measure how model size affects
throughput scaling and vLLM's optimization advantage.

Comparison targets:
  - Week 4/5 Llama 3.2 3B: 106 tok/s single, ~4,731 tok/s peak (HTTP)
  - Theoretical: 7B ~= half the per-request throughput of 3B
"""

import subprocess
import time
import asyncio
import aiohttp
import json
import os
import sys

# ── Configuration ──────────────────────────────────────────────
MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
GPU_ID = 0
PORT = 8000
MAX_MODEL_LEN = 4096

PROMPT = "Explain the concept of continuous batching in LLM inference and why it matters for production deployments."
OUTPUT_LEN = 128
WARMUP_REQUESTS = 5
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32, 64, 96]


def launch_vllm_server() -> subprocess.Popen:
    """Launch vLLM server for Mistral 7B on GPU 0."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(GPU_ID)}
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--port", str(PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--dtype", "float16",
        "--disable-log-requests",
        "--gpu-memory-utilization", "0.90",
    ]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


async def wait_for_server(timeout: int = 300):
    """Poll health endpoint until ready."""
    url = f"http://localhost:{PORT}/health"
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
    raise TimeoutError(f"Server did not start within {timeout}s")


async def send_request(session, port):
    """Send completion request, return (tokens, latency, tok/s)."""
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
    return tokens, elapsed, tokens / elapsed


async def benchmark(concurrency: int, num_requests: int):
    """Run benchmark at given concurrency, return aggregated results."""
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    timeout = aiohttp.ClientTimeout(total=180)

    results = []
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async def bounded():
            async with semaphore:
                return await send_request(session, PORT)
        tasks = [bounded() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
    return results


async def main():
    print(f"\n{'#'*70}")
    print(f"  Week 5 Experiment 2: Larger Model Benchmark")
    print(f"  Model: {MODEL}")
    print(f"  GPU: {GPU_ID} (RTX 3090, 24GB)")
    print(f"  Max sequence length: {MAX_MODEL_LEN}")
    print(f"{'#'*70}\n")

    # Launch server
    print("  Launching vLLM server...")
    proc = launch_vllm_server()
    print(f"  PID: {proc.pid}")

    try:
        print("  Waiting for model to load...")
        await wait_for_server()
        print("  Server ready.\n")

        # Warmup
        print(f"  Warming up ({WARMUP_REQUESTS} requests)...")
        await benchmark(1, WARMUP_REQUESTS)

        # Benchmark
        print(f"\n  {'Conc':>5}  {'Requests':>9}  {'Sys tok/s':>10}  "
              f"{'Avg Lat':>9}  {'Per-req tok/s':>14}  {'Wall Time':>10}")
        print(f"  {'----':>5}  {'--------':>9}  {'---------':>10}  "
              f"{'-------':>9}  {'-------------':>14}  {'---------':>10}")

        for conc in CONCURRENCY_LEVELS:
            num_req = max(conc * 3, 30)
            wall_start = time.time()
            results = await benchmark(conc, num_req)
            wall_time = time.time() - wall_start

            total_tokens = sum(r[0] for r in results)
            avg_latency = sum(r[1] for r in results) / len(results)
            avg_per_req = sum(r[2] for r in results) / len(results)
            sys_throughput = total_tokens / wall_time

            print(f"  {conc:>5}  {num_req:>9}  {sys_throughput:>9,.1f}  "
                  f"{avg_latency:>8.3f}s  {avg_per_req:>13.1f}  {wall_time:>9.2f}s")

        # Comparison
        print(f"\n{'='*70}")
        print(f"  Comparison (single GPU, same concurrency levels):")
        print(f"    Llama 3.2 3B @ conc=1:   106 tok/s per-request")
        print(f"    Llama 3.2 3B @ conc=64:  4,731 tok/s system, 74 tok/s per-request")
        print(f"{'='*70}\n")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
