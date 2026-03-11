#!/usr/bin/env python3
"""
Week 7 Experiment 1: Qwen 2.5 14B NVLink TP=2 Benchmark
Compares NVLink (GPU0+GPU2) tensor parallelism against Week 6 PCIe TP=2 baseline.

Usage:
    CUDA_VISIBLE_DEVICES=0,2 python3 exp1_qwen14b_nvlink_tp2_benchmark.py

Requires:
    - Qwen/Qwen2.5-14B-Instruct downloaded
    - vLLM installed in ai-inference venv
    - NVLink bridge connecting GPU0 and GPU2 (confirmed via nvidia-smi topo -m)
"""

import subprocess
import time
import json
import statistics
import os
import sys

# ── Enforce GPU selection ──────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

from vllm import LLM, SamplingParams

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
OUTPUT_TOKENS = 50
PROMPT = "Explain the difference between supervised and unsupervised learning in machine learning."
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
WARMUP_REQUESTS = 5

# ── Part A: Topology Verification ─────────────────────────────────


def print_topology():
    print("=" * 80)
    print("PART A: TOPOLOGY VERIFICATION")
    print("=" * 80)
    print()

    result = subprocess.run(
        ["nvidia-smi", "topo", "-m"], capture_output=True, text=True
    )
    print(result.stdout)

    # NVLink status per GPU
    for gpu_id in [0, 2]:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=name,uuid,pcie.link.gen.current,pcie.link.width.current",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        )
        print(f"  GPU {gpu_id}: {result.stdout.strip()}")

    print()
    print(
        "  Target pair: GPU0 <-> GPU2 via NV4 (4x NVLink lanes, ~100 GB/s bidirectional)"
    )
    print("  CUDA_VISIBLE_DEVICES: 0,2")
    print()


# ── Part B: NVLink Bandwidth Microbenchmark ────────────────────────


def nvlink_bandwidth_test():
    """Quick P2P bandwidth test between GPU0 and GPU2."""
    print("=" * 80)
    print("PART B: NVLink P2P BANDWIDTH MICROBENCHMARK")
    print("=" * 80)
    print()

    import torch

    # Check P2P access
    can_p2p = torch.cuda.can_device_access_peer(
        0, 1
    )  # device 0 = GPU0, device 1 = GPU2
    print(f"  P2P access GPU0 -> GPU2: {can_p2p}")

    # Allocate buffers
    sizes_mb = [32, 128, 512, 1024]
    for size_mb in sizes_mb:
        n_elements = (size_mb * 1024 * 1024) // 4  # float32
        src = torch.ones(n_elements, dtype=torch.float32, device="cuda:0")
        dst = torch.zeros(n_elements, dtype=torch.float32, device="cuda:1")

        # Warmup
        for _ in range(3):
            dst.copy_(src)
        torch.cuda.synchronize()

        # Benchmark
        trials = 10
        start = time.perf_counter()
        for _ in range(trials):
            dst.copy_(src)
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        bandwidth_gbs = (size_mb / 1024) * trials / elapsed
        print(f"  {size_mb:5d} MB transfer: {bandwidth_gbs:.2f} GB/s")

    del src, dst
    torch.cuda.empty_cache()
    print()


# ── Part C: vLLM TP=2 Inference Benchmark ─────────────────────────


def run_benchmark():
    print("=" * 80)
    print("PART C: QWEN 2.5 14B INFERENCE BENCHMARK (TP=2, NVLink)")
    print("=" * 80)
    print()

    print(f"  Model:       {MODEL_ID}")
    print(f"  TP degree:   2 (GPU0 + GPU2, NVLink)")
    print(f"  Output:      {OUTPUT_TOKENS} tokens/request")
    print(f"  Concurrency: {CONCURRENCY_LEVELS}")
    print()

    print("  Loading model...")
    t_load_start = time.time()
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=2,
        dtype="float16",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
    )
    t_load = time.time() - t_load_start
    print(f"  Model loaded in {t_load:.1f}s")
    print()

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=OUTPUT_TOKENS,
    )

    # Warmup
    print(f"  Warming up ({WARMUP_REQUESTS} requests)...")
    warmup_prompts = [PROMPT] * WARMUP_REQUESTS
    llm.generate(warmup_prompts, sampling_params)
    print("  Warmup complete.")
    print()

    results = []

    print(
        f"  {'Concurrency':>12} | {'Throughput':>14} | {'p50 Latency':>12} | {'p95 Latency':>12} | {'p99 Latency':>12}"
    )
    print(f"  {'-'*12}-+-{'-'*14}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for concurrency in CONCURRENCY_LEVELS:
        prompts = [PROMPT] * concurrency
        latencies = []

        # Run 3 trials at each concurrency level
        for trial in range(3):
            t_start = time.time()
            outputs = llm.generate(prompts, sampling_params)
            t_end = time.time()

            batch_time = t_end - t_start
            tokens_generated = sum(len(o.outputs[0].token_ids) for o in outputs)
            throughput = tokens_generated / batch_time
            latency_per_req = batch_time  # wall time for the batch

            latencies.append((throughput, latency_per_req, tokens_generated))

        throughputs = [l[0] for l in latencies]
        req_latencies = [l[1] for l in latencies]

        avg_throughput = statistics.mean(throughputs)
        p50 = statistics.median(req_latencies)
        p95 = (
            sorted(req_latencies)[int(len(req_latencies) * 0.95)]
            if len(req_latencies) >= 20
            else max(req_latencies)
        )
        p99 = (
            sorted(req_latencies)[int(len(req_latencies) * 0.99)]
            if len(req_latencies) >= 100
            else max(req_latencies)
        )

        results.append(
            {
                "concurrency": concurrency,
                "throughput_tok_s": round(avg_throughput, 1),
                "p50_latency_s": round(p50, 3),
                "p95_latency_s": round(p95, 3),
                "p99_latency_s": round(p99, 3),
            }
        )

        print(
            f"  {concurrency:>12} | {avg_throughput:>12.1f}/s | {p50:>10.3f}s | {p95:>10.3f}s | {p99:>10.3f}s"
        )

    return results


# ── Part D: PCIe Baseline Comparison ──────────────────────────────

# Week 6 results for direct comparison
WEEK6_PCIE_RESULTS = {
    1: {"throughput": 38.6, "p50": None},
    4: {"throughput": 96.2, "p50": None},
    16: {"throughput": 201.4, "p50": None},
    64: {"throughput": 278.3, "p50": None},
    256: {"throughput": 316.5, "p50": None},
}


def print_comparison(nvlink_results):
    print("=" * 80)
    print("PART D: NVLink vs PCIe COMPARISON")
    print("=" * 80)
    print()
    print(
        f"  {'Concurrency':>12} | {'PCIe tok/s':>12} | {'NVLink tok/s':>13} | {'Speedup':>8}"
    )
    print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*13}-+-{'-'*8}")

    for r in nvlink_results:
        c = r["concurrency"]
        nvlink_tput = r["throughput_tok_s"]
        pcie = WEEK6_PCIE_RESULTS.get(c)
        if pcie:
            pcie_tput = pcie["throughput"]
            speedup = nvlink_tput / pcie_tput
            print(
                f"  {c:>12} | {pcie_tput:>12.1f} | {nvlink_tput:>13.1f} | {speedup:>7.2f}x"
            )
        else:
            print(f"  {c:>12} | {'N/A':>12} | {nvlink_tput:>13.1f} | {'N/A':>8}")
    print()


# ── Main ───────────────────────────────────────────────────────────


def main():
    print()
    print("Week 7 Experiment 1: Qwen 2.5 14B NVLink TP=2 Benchmark")
    print("=" * 80)
    print()

    print_topology()
    nvlink_bandwidth_test()
    results = run_benchmark()
    print_comparison(results)

    # Save results
    output = {
        "experiment": "week7_exp1_nvlink_tp2",
        "model": MODEL_ID,
        "gpu_pair": "GPU0+GPU2 (NVLink NV4)",
        "tensor_parallel_size": 2,
        "output_tokens": OUTPUT_TOKENS,
        "results": results,
    }
    with open("results/nvlink_tp2_benchmark.txt", "w") as f:
        f.write(json.dumps(output, indent=2))

    print("  Results saved to results/nvlink_tp2_benchmark.txt")
    print()


if __name__ == "__main__":
    main()
