#!/usr/bin/env python3
"""
Week 7 Experiment 1b: Latency Distribution Sweep at Concurrency 128
Measures p50/p95/p99 latency distribution at peak throughput operating point.

Usage:
    python3 exp1b_latency_sweep.py

Assumes CUDA_VISIBLE_DEVICES=0,2 already set, or set internally below.
"""

import os
import time
import json
import statistics

os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

from vllm import LLM, SamplingParams

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
CONCURRENCY = 128
TRIALS = 30
WARMUP = 5
PROMPT = "Explain the difference between supervised and unsupervised learning in machine learning."

OUTPUT_TOKEN_SWEEP = [25, 50, 100, 200, 400]


def load_model():
    print("Loading model...")
    t0 = time.time()
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=2,
        dtype="float16",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s\n")
    return llm


def run_latency_distribution(llm):
    """30 repeated trials at concurrency=128, output=50 tokens.
    Produces a real p95/p99 from wall-time per batch."""

    print("=" * 70)
    print("PART A: LATENCY DISTRIBUTION — 30 trials at concurrency 128")
    print("=" * 70)
    print()

    sampling_params = SamplingParams(temperature=0.0, max_tokens=50)
    prompts = [PROMPT] * CONCURRENCY

    # Warmup
    print(f"  Warming up ({WARMUP} trials)...")
    for _ in range(WARMUP):
        llm.generate(prompts, sampling_params)
    print("  Done.\n")

    batch_times = []
    throughputs = []

    for i in range(TRIALS):
        t0 = time.time()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.time() - t0

        tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        batch_times.append(elapsed)
        throughputs.append(tokens / elapsed)

        print(f"  Trial {i+1:2d}: {elapsed:.3f}s  |  {tokens/elapsed:,.0f} tok/s")

    batch_times.sort()
    p50 = statistics.median(batch_times)
    p95 = batch_times[int(TRIALS * 0.95)]
    p99 = batch_times[int(TRIALS * 0.99)] if TRIALS >= 100 else batch_times[-1]
    mean = statistics.mean(batch_times)
    stdev = statistics.stdev(batch_times)

    print()
    print(f"  Mean latency:   {mean:.3f}s")
    print(f"  Stdev:          {stdev:.3f}s")
    print(f"  p50:            {p50:.3f}s")
    print(f"  p95:            {p95:.3f}s")
    print(f"  p99:            {batch_times[-1]:.3f}s  (worst of {TRIALS})")
    print(f"  Min:            {batch_times[0]:.3f}s")
    print(f"  Max:            {batch_times[-1]:.3f}s")
    print(f"  Mean throughput:{statistics.mean(throughputs):,.0f} tok/s")
    print()

    return {
        "trial_times": batch_times,
        "mean_s": round(mean, 3),
        "stdev_s": round(stdev, 3),
        "p50_s": round(p50, 3),
        "p95_s": round(p95, 3),
        "p99_s": round(batch_times[-1], 3),
        "min_s": round(batch_times[0], 3),
        "max_s": round(batch_times[-1], 3),
        "mean_throughput_tok_s": round(statistics.mean(throughputs), 1),
    }


def run_output_length_sweep(llm):
    """Sweep output token counts at concurrency=128 to measure latency scaling."""

    print("=" * 70)
    print("PART B: OUTPUT LENGTH SWEEP at concurrency 128")
    print("=" * 70)
    print()
    print(
        f"  {'Tokens':>8} | {'Latency (s)':>12} | {'Throughput':>14} | {'tok/s/token':>12}"
    )
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*14}-+-{'-'*12}")

    sweep_results = []

    for n_tokens in OUTPUT_TOKEN_SWEEP:
        sampling_params = SamplingParams(temperature=0.0, max_tokens=n_tokens)
        prompts = [PROMPT] * CONCURRENCY

        # 3 trials, take median
        times = []
        for _ in range(3):
            t0 = time.time()
            outputs = llm.generate(prompts, sampling_params)
            times.append(time.time() - t0)

        times.sort()
        median_time = times[1]
        tokens_generated = CONCURRENCY * n_tokens
        throughput = tokens_generated / median_time
        tok_s_per_output_token = throughput / n_tokens

        sweep_results.append(
            {
                "output_tokens": n_tokens,
                "latency_s": round(median_time, 3),
                "throughput_tok_s": round(throughput, 1),
            }
        )

        print(
            f"  {n_tokens:>8} | {median_time:>12.3f} | {throughput:>12,.0f}/s | {tok_s_per_output_token:>10.1f}"
        )

    print()
    return sweep_results


def main():
    print()
    print("Week 7 Exp 1b: Latency Sweep at Concurrency 128 (NVLink TP=2)")
    print("=" * 70)
    print()

    llm = load_model()

    dist_results = run_latency_distribution(llm)
    sweep_results = run_output_length_sweep(llm)

    output = {
        "experiment": "week7_exp1b_latency_sweep",
        "model": MODEL_ID,
        "gpu_pair": "GPU0+GPU2 (NVLink NV4)",
        "concurrency": CONCURRENCY,
        "latency_distribution": dist_results,
        "output_length_sweep": sweep_results,
    }

    with open("results/nvlink_latency_sweep.txt", "w") as f:
        f.write(json.dumps(output, indent=2))

    print("Results saved to results/nvlink_latency_sweep.txt")


if __name__ == "__main__":
    main()
