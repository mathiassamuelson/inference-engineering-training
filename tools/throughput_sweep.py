#!/usr/bin/env python3
"""
throughput_sweep.py — single-request throughput sweep for OpenAI-compatible LLM endpoints.

Supports vLLM (native OpenAI-compatible) and llama.cpp-server (OpenAI-compatible mode).
Uses streaming completions to measure per-request prefill and decode rates, and emits a
self-describing JSON results file.

Timing methodology:
  - Wall clock is captured around the HTTP request.
  - Time-to-first-token (TTFT) is the wall-clock from request send to the first chunk
    containing non-empty text. This is treated as the prefill window.
  - Token counts come from the final chunk's `usage` block (requires
    `stream_options.include_usage=true`, which vLLM supports natively; llama.cpp-server
    compatibility should be validated before trusting results from that backend).
  - Decode rate is (completion_tokens - 1) / (wall_time - ttft): the first generated
    token is produced at TTFT, so only the remaining tokens belong to the decode window.

Backend-agnostic by construction: the same request/response path is used for both
backends. The `--backend` flag is recorded in results metadata for provenance, not to
switch code paths.
"""

import argparse
import json
import platform
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Optional

import requests


BACKENDS = ("llamacpp", "vllm-openai")
SCHEMA_VERSION = 1


def get_git_info() -> dict:
    """Return git SHA and dirty-tree status for the current working directory."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip())
        return {"git_sha": sha, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_sha": None, "dirty": None}


def slugify_model_name(name: str) -> str:
    """Make a model name safe for use in filenames."""
    name = name.split("/")[-1]  # strip org prefix if present
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in name)


def discover_model_name(endpoint: str, timeout: float = 5.0) -> Optional[str]:
    """Query /v1/models and return the first model id, or None on failure."""
    url = endpoint.rstrip("/") + "/v1/models"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", [])
        if not models:
            return None
        return models[0].get("id")
    except (requests.RequestException, ValueError):
        return None


def build_prompt(approx_tokens: int) -> str:
    """
    Build a prompt of approximately the requested token count.

    Uses a simple repeating filler. The actual token count is captured from the
    server response in each result record — we care about the server's reported
    prompt_tokens, not what we tried to send.
    """
    word = "lorem "
    words_needed = max(1, int(approx_tokens / 0.75))
    return (word * words_needed).strip()


def run_streaming_request(
    endpoint: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> dict:
    """
    Issue a streaming completion request and return per-request timing + token counts.

    Raises RuntimeError if the response stream does not include a usage block.
    """
    url = endpoint.rstrip("/") + "/v1/completions"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.perf_counter()
    t_first_token = None
    usage = None

    with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                line = line[len("data: "):]
            if line.strip() == "[DONE]":
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            if chunk.get("usage"):
                usage = chunk["usage"]

            choices = chunk.get("choices") or []
            if choices and t_first_token is None:
                text = choices[0].get("text", "")
                if text:
                    t_first_token = time.perf_counter()

    t_end = time.perf_counter()

    if usage is None:
        raise RuntimeError(
            "No usage block in streamed response. "
            "Backend may not support stream_options.include_usage."
        )

    prompt_tokens = usage["prompt_tokens"]
    completion_tokens = usage["completion_tokens"]

    wall_time = t_end - t_start
    if t_first_token is None:
        # No text chunks observed — degenerate case, treat entire wall time as prefill.
        ttft = wall_time
    else:
        ttft = t_first_token - t_start

    prefill_time = ttft
    decode_time = max(wall_time - ttft, 1e-9)

    prefill_rate = prompt_tokens / prefill_time if prefill_time > 0 else 0.0
    # Subtract 1: the first decoded token is produced at TTFT, not during decode.
    decode_rate = max(completion_tokens - 1, 0) / decode_time

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "wall_time_s": wall_time,
        "ttft_s": ttft,
        "prefill_time_s": prefill_time,
        "decode_time_s": decode_time,
        "prefill_rate_tok_s": prefill_rate,
        "decode_rate_tok_s": decode_rate,
    }


def summarize(records: list) -> dict:
    """Compute summary statistics across iterations for the key timing fields."""
    def stat(key):
        values = [r[key] for r in records]
        return {
            "mean": mean(values),
            "median": median(values),
            "min": min(values),
            "max": max(values),
            "stdev": stdev(values) if len(values) > 1 else 0.0,
        }
    return {
        "wall_time_s": stat("wall_time_s"),
        "ttft_s": stat("ttft_s"),
        "prefill_rate_tok_s": stat("prefill_rate_tok_s"),
        "decode_rate_tok_s": stat("decode_rate_tok_s"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Single-request throughput sweep for OpenAI-compatible LLM endpoints.",
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=BACKENDS,
        help="Backend serving the endpoint. Recorded in results metadata for provenance.",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000",
        help="Base URL of the OpenAI-compatible server (default: %(default)s).",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model identifier. If omitted, queried from /v1/models; exits if discovery fails.",
    )
    parser.add_argument(
        "--prompt-sizes",
        type=int,
        nargs="+",
        default=[128, 512, 1024, 2048, 4096],
        help="Approximate prompt sizes in tokens to sweep (default: %(default)s). "
             "Actual counts come from the server.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Completion tokens requested per call (default: %(default)s).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Measured iterations per prompt size (default: %(default)s).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup iterations per prompt size, discarded (default: %(default)s).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=600.0,
        help="Per-request timeout in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file. Default: "
             "<results-dir>/throughput_sweep_<backend>_<model>_<timestamp>.json",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for default output filename (default: %(default)s).",
    )
    args = parser.parse_args()

    # Resolve model name (explicit or discovered)
    model_source = "explicit"
    model_name = args.model_name
    if model_name is None:
        print(
            f"[info] --model-name not specified; querying {args.endpoint}/v1/models",
            file=sys.stderr,
        )
        model_name = discover_model_name(args.endpoint)
        if model_name is None:
            print(
                "[error] Could not discover model name from /v1/models. "
                "Pass --model-name explicitly.",
                file=sys.stderr,
            )
            sys.exit(2)
        model_source = "discovered"
        print(f"[info] discovered model: {model_name}", file=sys.stderr)

    # Resolve output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = slugify_model_name(model_name)
        filename = f"throughput_sweep_{args.backend}_{slug}_{timestamp}.json"
        output_path = Path(args.results_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run header
    print(f"throughput_sweep: backend={args.backend} model={model_name}", file=sys.stderr)
    print(f"  endpoint={args.endpoint}", file=sys.stderr)
    print(f"  prompt_sizes={args.prompt_sizes}  max_tokens={args.max_tokens}", file=sys.stderr)
    print(f"  iterations={args.iterations}  warmup={args.warmup}", file=sys.stderr)
    print(f"  output={output_path}", file=sys.stderr)
    print("", file=sys.stderr)

    # Sweep
    results = []
    for size in args.prompt_sizes:
        print(f"[prompt_size={size}]", file=sys.stderr)
        prompt = build_prompt(size)

        for i in range(args.warmup):
            try:
                run_streaming_request(
                    args.endpoint, model_name, prompt, args.max_tokens, args.request_timeout,
                )
                print(f"  warmup {i+1}/{args.warmup} ok", file=sys.stderr)
            except Exception as e:
                print(f"  warmup {i+1}/{args.warmup} FAILED: {e}", file=sys.stderr)

        iters = []
        for i in range(args.iterations):
            try:
                rec = run_streaming_request(
                    args.endpoint, model_name, prompt, args.max_tokens, args.request_timeout,
                )
                iters.append(rec)
                print(
                    f"  iter {i+1}/{args.iterations}: "
                    f"prompt={rec['prompt_tokens']}tok "
                    f"gen={rec['completion_tokens']}tok "
                    f"prefill={rec['prefill_rate_tok_s']:.1f}tok/s "
                    f"decode={rec['decode_rate_tok_s']:.1f}tok/s",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"  iter {i+1}/{args.iterations} FAILED: {e}", file=sys.stderr)

        entry = {
            "prompt_size_requested": size,
            "iterations": iters,
        }
        if iters:
            entry["summary"] = summarize(iters)
        results.append(entry)

    # Metadata (no model-specific hardcoded fields)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "script": {
            "name": "throughput_sweep.py",
            "git": get_git_info(),
        },
        "run": {
            "run_id": str(uuid.uuid4()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        },
        "backend": args.backend,
        "endpoint": args.endpoint,
        "model": {
            "name": model_name,
            "source": model_source,
        },
        "sweep_config": {
            "prompt_sizes_requested": args.prompt_sizes,
            "max_tokens": args.max_tokens,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "request_timeout_s": args.request_timeout,
        },
    }

    output = {
        "metadata": metadata,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
