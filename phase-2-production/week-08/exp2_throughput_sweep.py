#!/usr/bin/env python3
"""
Week 8 Experiment 2: Throughput Sweep — Gemma 4 on NVLink (llama.cpp)

Measures prompt processing speed and generation speed at various context lengths.
Sends synthetic prompts of increasing size to the llama-server OpenAI-compatible API
and captures timing data from the response.

Usage:
    python3 exp2_throughput_sweep.py [--host HOST] [--port PORT] [--output-tokens N]
                                     [--model-name NAME]

Hardware: 2x RTX 3090 (NVLink, GPU 0+2), layer splitting via llama.cpp
Default model: ggml-org/gemma-4-31B-it-GGUF Q8_0
"""

import argparse
import json
import time
import requests
import sys
import uuid

# --- Configuration ---

API_BASE = "http://localhost:8080"
OUTPUT_TOKENS = 100  # Fixed output length for fair comparison
DEFAULT_MODEL_NAME = "gemma-4-31B-it"

# Target prompt sizes in tokens (approximate — actual count comes from API response)
# We overshoot slightly since token/word ratio varies; the API reports exact counts
TARGET_PROMPT_TOKENS = [500, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000]

# Filler sentence (~20 tokens) repeated to build prompts of target sizes
FILLER_SENTENCE = "The quick brown fox jumps over the lazy dog near the river bank. "
TOKENS_PER_FILLER = 16  # Approximate tokens per filler sentence


def build_prompt(target_tokens: int) -> str:
    repeats = max(1, target_tokens // TOKENS_PER_FILLER)
    filler = FILLER_SENTENCE * repeats
    nonce = uuid.uuid4().hex  # unique per call, ~32 tokens
    prompt = (
        f"[session {nonce}] "
        f"I am going to give you a block of repeated text. "
        f"After reading it, write a single paragraph summarizing what you observed. "
        f"Here is the text:\n\n{filler}\n\n"
        f"Now write your one-paragraph summary."
    )
    return prompt


def send_request(prompt: str, max_tokens: int, api_base: str, model_name: str) -> dict:
    """Send a chat completion request and return the full response."""
    url = f"{api_base}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
    }

    start = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "wall_time": time.time() - start}

    wall_time = time.time() - start
    data = resp.json()
    data["wall_time"] = wall_time
    return data


def extract_timings(response: dict) -> dict:
    """Extract timing metrics from llama.cpp response."""
    if "error" in response:
        return {"error": response["error"]}

    timings = response.get("timings", {})
    usage = response.get("usage", {})

    return {
        "prompt_tokens": timings.get("prompt_tokens", usage.get("prompt_tokens", 0)),
        "generated_tokens": timings.get(
            "predicted_n", usage.get("completion_tokens", 0)
        ),
        "prompt_ms": timings.get("prompt_ms", 0),
        "generation_ms": timings.get("predicted_ms", 0),
        "prompt_tok_s": timings.get("prompt_per_second", 0),
        "generation_tok_s": timings.get("predicted_per_second", 0),
        "wall_time_s": response.get("wall_time", 0),
        "finish_reason": (
            response["choices"][0]["finish_reason"]
            if response.get("choices")
            else "unknown"
        ),
    }


def run_warmup(api_base: str, model_name: str):
    """Send a short warmup request to prime CUDA graphs and caches."""
    print("Warming up...", flush=True)
    prompt = "Say hello in one sentence."
    resp = send_request(prompt, max_tokens=20, api_base=api_base, model_name=model_name)
    if "error" in resp:
        print(f"  Warmup failed: {resp['error']}")
        sys.exit(1)
    timings = extract_timings(resp)
    print(
        f"  Warmup complete: {timings['generated_tokens']} tokens in {timings['wall_time_s']:.1f}s\n"
    )


def run_sweep(api_base: str, output_tokens: int, targets: list, model_name: str):
    """Run the throughput sweep across target prompt sizes."""
    results = []

    for target in targets:
        print(f"Testing ~{target:,} prompt tokens...", end=" ", flush=True)

        prompt = build_prompt(target)
        response = send_request(
            prompt, max_tokens=output_tokens, api_base=api_base, model_name=model_name
        )
        metrics = extract_timings(response)

        if "error" in metrics:
            print(f"FAILED: {metrics['error']}")
            results.append({"target": target, "error": metrics["error"]})
            continue

        actual_prompt = metrics["prompt_tokens"]
        print(
            f"actual={actual_prompt:,} prompt tokens | "
            f"prefill={metrics['prompt_tok_s']:.1f} tok/s | "
            f"decode={metrics['generation_tok_s']:.1f} tok/s | "
            f"wall={metrics['wall_time_s']:.1f}s"
        )

        results.append(
            {
                "target_prompt_tokens": target,
                "actual_prompt_tokens": actual_prompt,
                "generated_tokens": metrics["generated_tokens"],
                "prompt_ms": metrics["prompt_ms"],
                "generation_ms": metrics["generation_ms"],
                "prompt_tok_s": metrics["prompt_tok_s"],
                "generation_tok_s": metrics["generation_tok_s"],
                "wall_time_s": metrics["wall_time_s"],
                "finish_reason": metrics["finish_reason"],
            }
        )

    return results


def print_summary(results: list, output_tokens: int, model_name: str):
    """Print a formatted summary table."""
    print()
    print("=" * 110)
    print(
        f"THROUGHPUT SWEEP RESULTS — {model_name} Q8_0 | 2x RTX 3090 NVLink | llama.cpp layer splitting"
    )
    print(f"Output tokens per request: {output_tokens}")
    print("=" * 110)
    print(
        f"{'Prompt Tokens':>14} | "
        f"{'Prefill (tok/s)':>16} | "
        f"{'Decode (tok/s)':>15} | "
        f"{'Prefill Time':>13} | "
        f"{'Decode Time':>12} | "
        f"{'Wall Time':>10} | "
        f"{'Status':>8}"
    )
    print("-" * 110)

    for r in results:
        if "error" in r:
            print(f"{r['target']:>14,} | {'FAILED':>16} |")
            continue

        print(
            f"{r['actual_prompt_tokens']:>14,} | "
            f"{r['prompt_tok_s']:>16.1f} | "
            f"{r['generation_tok_s']:>15.1f} | "
            f"{r['prompt_ms']/1000:>12.2f}s | "
            f"{r['generation_ms']/1000:>11.2f}s | "
            f"{r['wall_time_s']:>9.2f}s | "
            f"{r['finish_reason']:>8}"
        )

    print("=" * 110)


def save_results(results: list, output_tokens: int, filepath: str, model_name: str):
    """Save raw results to JSON."""
    output = {
        "experiment": "week8_exp2_throughput_sweep",
        "model": f"{model_name} Q8_0",
        "hardware": "2x RTX 3090 NVLink (GPU 0+2), layer splitting",
        "framework": "llama.cpp",
        "output_tokens_per_request": output_tokens,
        "kv_cache_type": "f16",
        "results": results,
    }
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Gemma 4 throughput sweep")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--output-tokens", type=int, default=OUTPUT_TOKENS)
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Model name used in headers, JSON metadata, and default output filename",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output JSON path. Defaults to results/exp2_throughput_sweep_<model-name>.json",
    )
    args = parser.parse_args()

    api_base = f"http://{args.host}:{args.port}"
    output_file = (
        args.output_file
        or f"results/exp2_throughput_sweep_{args.model_name}.json"
    )

    print(f"{args.model_name} Q8_0 — Throughput Sweep")
    print(f"API: {api_base}")
    print(f"Output tokens per request: {args.output_tokens}")
    print(f"Target prompt sizes: {[f'{t:,}' for t in TARGET_PROMPT_TOKENS]}")
    print()

    run_warmup(api_base, args.model_name)
    results = run_sweep(
        api_base, args.output_tokens, TARGET_PROMPT_TOKENS, args.model_name
    )
    print_summary(results, args.output_tokens, args.model_name)
    save_results(results, args.output_tokens, output_file, args.model_name)


if __name__ == "__main__":
    main()
