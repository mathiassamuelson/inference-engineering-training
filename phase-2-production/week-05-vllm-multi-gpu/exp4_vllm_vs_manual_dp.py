"""
Week 5 - Experiment 4: vLLM vs Manual Data Parallelism
Direct comparison using identical sustained mixed workloads.

Part A: transformers + multiprocessing (replicating Week 3 approach)
Part B: vLLM multi-instance (replicating Experiment 3 approach)
Same concurrency, same duration, same request mix.
"""

import subprocess
import time
import asyncio
import aiohttp
import random
import os
import sys
import signal
import statistics
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from queue import Empty
from typing import Optional

# ── Configuration ──────────────────────────────────────────────
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
GPU_IDS = [0, 1, 2, 3]
BASE_PORT = 8000
MAX_MODEL_LEN = 4096

TEST_DURATION_S = 60
CONCURRENCY = 32  # total across all GPUs

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
    "Summarize the key differences between data parallelism and tensor parallelism.",
    "What is continuous batching and why does it matter?",
    "Explain the concept of KV cache in autoregressive language models.",
]


@dataclass
class RequestResult:
    gpu_id: int
    tokens_generated: int
    latency_s: float
    profile_name: str
    timestamp: float


def pick_mixed_profile() -> dict:
    r = random.random()
    cumulative = 0.0
    for profile in MIXED_PROFILES:
        cumulative += profile["weight"]
        if r <= cumulative:
            return profile
    return MIXED_PROFILES[-1]


# ── Part A: Transformers + Multiprocessing ─────────────────────


def transformers_worker(
    gpu_id: int, request_queue: mp.Queue, result_queue: mp.Queue, stop_event
):
    """Worker process: loads model on one GPU, processes requests from queue."""
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map=device
    )
    model.eval()

    # Signal ready
    result_queue.put(("ready", gpu_id))

    while not stop_event.is_set():
        try:
            req = request_queue.get(timeout=0.1)
        except Empty:
            continue

        if req is None:  # poison pill
            break

        prompt_text, max_tokens, profile_name = req
        start = time.time()

        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
        elapsed = time.time() - start

        result_queue.put(
            RequestResult(
                gpu_id=gpu_id,
                tokens_generated=tokens_generated,
                latency_s=elapsed,
                profile_name=profile_name,
                timestamp=time.time(),
            )
        )


def run_transformers_test(duration_s: float, concurrency: int) -> list[RequestResult]:
    """
    Run transformers-based data parallel inference.
    Uses a shared request queue with N worker processes (one per GPU).
    Each worker pulls one request at a time — this is static batching
    with batch_size=1, the closest analogue to how a naive deployment works.
    """

    request_queue = mp.Queue()
    result_queue = mp.Queue()
    stop_event = mp.Event()

    # Launch workers
    workers = []
    for gpu_id in GPU_IDS:
        p = mp.Process(
            target=transformers_worker,
            args=(gpu_id, request_queue, result_queue, stop_event),
        )
        p.start()
        workers.append(p)

    # Wait for all workers to be ready
    ready_count = 0
    while ready_count < len(GPU_IDS):
        msg = result_queue.get(timeout=300)
        if msg[0] == "ready":
            ready_count += 1
            print(f"    GPU {msg[1]} ready")

    print(f"    All {len(GPU_IDS)} workers ready.\n")

    # Pre-fill the queue to maintain concurrency
    # With batch_size=1 per worker, we can have at most len(GPU_IDS) in-flight
    # but we fill the queue to keep workers busy
    results = []
    end_time = time.time() + duration_s
    submitted = 0

    # Fill queue with initial batch
    for _ in range(concurrency):
        profile = pick_mixed_profile()
        prompt = random.choice(PROMPTS)
        request_queue.put((prompt, profile["max_tokens"], profile["name"]))
        submitted += 1

    # Collect results and submit replacements
    while time.time() < end_time:
        try:
            result = result_queue.get(timeout=0.5)
            if isinstance(result, RequestResult):
                results.append(result)
                # Submit replacement
                if time.time() < end_time:
                    profile = pick_mixed_profile()
                    prompt = random.choice(PROMPTS)
                    request_queue.put((prompt, profile["max_tokens"], profile["name"]))
                    submitted += 1
        except Empty:
            continue

    # Stop workers
    stop_event.set()
    for _ in workers:
        request_queue.put(None)

    # Drain remaining results briefly
    drain_deadline = time.time() + 5
    while time.time() < drain_deadline:
        try:
            result = result_queue.get(timeout=0.5)
            if isinstance(result, RequestResult):
                results.append(result)
        except Empty:
            break

    for p in workers:
        p.join(timeout=10)
        if p.is_alive():
            p.kill()

    return results


# ── Part B: vLLM Multi-Instance ────────────────────────────────


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
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
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


async def vllm_send_request(session, port, gpu_id, max_tokens, profile_name):
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
        profile_name=profile_name,
        timestamp=time.time(),
    )


async def run_vllm_test(
    duration_s: float, concurrency: int
) -> tuple[list[RequestResult], list]:
    """Run vLLM sustained mixed workload — same as Experiment 3 Phase B."""
    servers = []
    ports = []

    for gpu_id in GPU_IDS:
        port = BASE_PORT + gpu_id
        proc = launch_vllm_server(gpu_id, port)
        servers.append(proc)
        ports.append(port)
        print(f"    GPU {gpu_id} → port {port} (pid {proc.pid})")

    print(f"\n    Waiting for servers...")
    await asyncio.gather(*[wait_for_server(p) for p in ports])
    print(f"    All servers ready.\n")

    # Warmup
    connector = aiohttp.TCPConnector(limit=20)
    async with aiohttp.ClientSession(connector=connector) as session:
        warmup_tasks = []
        for port, gpu_id in zip(ports, GPU_IDS):
            for _ in range(3):
                warmup_tasks.append(
                    vllm_send_request(session, port, gpu_id, 64, "warmup")
                )
        await asyncio.gather(*warmup_tasks)

    # Sustained load
    results = []
    end_time = time.time() + duration_s
    num_gpus = len(ports)
    request_counter = 0

    connector = aiohttp.TCPConnector(limit=concurrency + 20)
    timeout = aiohttp.ClientTimeout(total=120)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        active_tasks: set[asyncio.Task] = set()

        def make_request():
            nonlocal request_counter
            idx = request_counter % num_gpus
            request_counter += 1
            profile = pick_mixed_profile()
            return asyncio.create_task(
                vllm_send_request(
                    session,
                    ports[idx],
                    GPU_IDS[idx],
                    profile["max_tokens"],
                    profile["name"],
                )
            )

        for _ in range(concurrency):
            if time.time() >= end_time:
                break
            active_tasks.add(make_request())

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
                    results.append(task.result())
                except Exception:
                    pass
                if time.time() < end_time:
                    active_tasks.add(make_request())

        for task in active_tasks:
            task.cancel()
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)

    return results, servers


def analyze_results(results: list[RequestResult], phase_name: str, wall_time: float):
    """Print analysis matching Experiment 3 format."""
    if not results:
        print(f"  No results for {phase_name}")
        return

    latencies = sorted([r.latency_s for r in results])
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
    print(f"    p50:  {latencies[len(latencies)//2]:.3f}s")
    print(f"    p95:  {latencies[int(len(latencies)*0.95)]:.3f}s")
    print(f"    p99:  {latencies[int(len(latencies)*0.99)]:.3f}s")
    print(f"    min:  {latencies[0]:.3f}s")
    print(f"    max:  {latencies[-1]:.3f}s")

    # Per-GPU breakdown
    per_gpu = {}
    for r in results:
        per_gpu.setdefault(r.gpu_id, []).append(r)

    print()
    print(f"  {'GPU':>5}  {'Reqs':>6}  {'Tokens':>8}  {'Avg Lat':>9}  {'tok/s':>7}")
    print(f"  {'---':>5}  {'----':>6}  {'------':>8}  {'-------':>9}  {'-----':>7}")
    for gid in sorted(per_gpu.keys()):
        gr = per_gpu[gid]
        gt = sum(r.tokens_generated for r in gr)
        gl = statistics.mean([r.latency_s for r in gr])
        gtps = sum(r.tokens_generated / r.latency_s for r in gr) / len(gr)
        print(f"  {gid:>5}  {len(gr):>6}  {gt:>8,}  {gl:>8.3f}s  {gtps:>6.1f}")


async def async_main():
    print(f"\n{'#'*70}")
    print(f"  Week 5 Experiment 4: vLLM vs Manual Data Parallelism")
    print(f"  Model: {MODEL}")
    print(f"  GPUs: {len(GPU_IDS)} × RTX 3090")
    print(f"  Concurrency: {CONCURRENCY} | Duration: {TEST_DURATION_S}s each")
    print(f"  Workload: Mixed (32-512 tokens, weighted)")
    print(f"{'#'*70}")

    # ── Part A: Transformers ───────────────────────────────────
    print(f"\n  Part A: Transformers + Multiprocessing")
    print(f"  {'─'*50}")
    print(f"  Launching {len(GPU_IDS)} worker processes...\n")

    wall_start = time.time()
    tf_results = run_transformers_test(TEST_DURATION_S, CONCURRENCY)
    tf_wall = time.time() - wall_start

    analyze_results(tf_results, "Part A: Transformers Data Parallel", tf_wall)

    # Brief cooldown
    print(f"\n  Cooling down (10s)...")
    await asyncio.sleep(10)

    # ── Part B: vLLM ───────────────────────────────────────────
    print(f"\n  Part B: vLLM Multi-Instance")
    print(f"  {'─'*50}")

    vllm_servers = []
    try:
        wall_start = time.time()
        vllm_results, vllm_servers = await run_vllm_test(TEST_DURATION_S, CONCURRENCY)
        vllm_wall = time.time() - wall_start

        analyze_results(vllm_results, "Part B: vLLM Data Parallel", vllm_wall)
    finally:
        cleanup_vllm_servers(vllm_servers)

    # ── Comparison ─────────────────────────────────────────────
    print(f"\n\n{'#'*70}")
    print(f"  Direct Comparison Summary")
    print(f"{'#'*70}")

    if tf_results and vllm_results:
        tf_tps = sum(r.tokens_generated for r in tf_results) / tf_wall
        vl_tps = sum(r.tokens_generated for r in vllm_results) / vllm_wall
        tf_rps = len(tf_results) / tf_wall
        vl_rps = len(vllm_results) / vllm_wall
        tf_avg_lat = statistics.mean([r.latency_s for r in tf_results])
        vl_avg_lat = statistics.mean([r.latency_s for r in vllm_results])

        print(f"\n  {'Metric':<25}  {'Transformers':>14}  {'vLLM':>14}  {'Ratio':>8}")
        print(f"  {'-'*24:<25}  {'-'*13:>14}  {'-'*13:>14}  {'-'*7:>8}")
        print(f"  {'System throughput':<25}  {tf_tps:>12,.1f}  {vl_tps:>12,.1f}  {vl_tps/tf_tps:>7.2f}x")
        print(f"  {'Request rate':<25}  {tf_rps:>12.1f}  {vl_rps:>12.1f}  {vl_rps/tf_rps:>7.2f}x")
        print(f"  {'Avg latency':<25}  {tf_avg_lat:>11.3f}s  {vl_avg_lat:>11.3f}s  {tf_avg_lat/vl_avg_lat:>7.2f}x")
        print(f"  {'Total requests':<25}  {len(tf_results):>14}  {len(vllm_results):>14}")

        # Per-profile comparison
        print(f"\n  Per-profile latency comparison:")
        print(f"  {'Profile':<16}  {'TF Lat':>9}  {'vLLM Lat':>9}  {'Speedup':>8}")
        print(f"  {'-'*15:<16}  {'-'*8:>9}  {'-'*8:>9}  {'-'*7:>8}")
        for name in ["quick_reply", "short_answer", "explanation", "long_response"]:
            tf_prof = [r for r in tf_results if r.profile_name == name]
            vl_prof = [r for r in vllm_results if r.profile_name == name]
            if tf_prof and vl_prof:
                tf_lat = statistics.mean([r.latency_s for r in tf_prof])
                vl_lat = statistics.mean([r.latency_s for r in vl_prof])
                print(f"  {name:<16}  {tf_lat:>8.3f}s  {vl_lat:>8.3f}s  {tf_lat/vl_lat:>7.2f}x")

    print(f"\n  Week 3 reference: 7,422 tok/s (batch=32, uniform, transformers)")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    print("Starting...", flush=True)
    mp.set_start_method("spawn", force=True)
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        import traceback

        traceback.print_exc()
        sys.exit(1)
