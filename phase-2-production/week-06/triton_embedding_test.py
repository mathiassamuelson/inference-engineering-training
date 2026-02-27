#!/usr/bin/env python3
"""
Week 6 Experiment 3: Triton Inference Server — Embedding Model Test
Validates deployment, tests inference, benchmarks dynamic batching.

Triton HTTP API (mapped to port 8001):
  GET  /v2/health/ready          — server readiness
  GET  /v2/models/embedding      — model metadata
  POST /v2/models/embedding/infer — inference request

Usage:
  1. Start Triton: ./start_triton.sh
  2. Run test:     python3 triton_embedding_test.py
"""

import requests
import numpy as np
import json
import time
import asyncio
import aiohttp
import statistics
from transformers import AutoTokenizer

TRITON_URL = "http://localhost:8001"
MODEL_NAME = "embedding"
TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ── Part A: Health & Metadata ──────────────────────────────────────

def check_health():
    """Verify Triton server and model are ready."""
    print("=" * 80)
    print("PART A: TRITON HEALTH & METADATA")
    print("=" * 80)
    print()

    # Server health
    resp = requests.get(f"{TRITON_URL}/v2/health/ready")
    print(f"  Server ready:  {resp.status_code == 200} (HTTP {resp.status_code})")

    # Model metadata
    resp = requests.get(f"{TRITON_URL}/v2/models/{MODEL_NAME}")
    meta = resp.json()
    print(f"  Model name:    {meta['name']}")
    print(f"  Versions:      {meta.get('versions', 'N/A')}")
    print(f"  Platform:      {meta.get('platform', 'N/A')}")
    print()

    # Model config
    resp = requests.get(f"{TRITON_URL}/v2/models/{MODEL_NAME}/config")
    config = resp.json()
    print(f"  Max batch:     {config.get('max_batch_size', 'N/A')}")
    print(f"  Inputs:        {[inp['name'] for inp in config.get('input', [])]}")
    print(f"  Outputs:       {[out['name'] for out in config.get('output', [])]}")
    print(f"  Instance:      GPU {config.get('instance_group', [{}])[0].get('gpus', 'N/A')}")

    db = config.get('dynamic_batching', {})
    if db:
        print(f"  Dyn batching:  preferred={db.get('preferred_batch_size', [])}, "
              f"max_delay={db.get('max_queue_delay_microseconds', 0)}μs")
    print()
    return True


# ── Part B: Single Inference ───────────────────────────────────────

def single_inference(tokenizer):
    """Send a single embedding request and validate the output."""
    print("=" * 80)
    print("PART B: SINGLE INFERENCE REQUEST")
    print("=" * 80)
    print()

    text = "GPU memory bandwidth determines inference throughput"
    encoded = tokenizer(text, padding="max_length", max_length=128,
                        truncation=True, return_tensors="np")

    # Build Triton inference request
    payload = {
        "inputs": [
            {
                "name": "input_ids",
                "shape": [1, 128],
                "datatype": "INT64",
                "data": encoded["input_ids"].tolist()
            },
            {
                "name": "attention_mask",
                "shape": [1, 128],
                "datatype": "INT64",
                "data": encoded["attention_mask"].tolist()
            },
            {
                "name": "token_type_ids",
                "shape": [1, 128],
                "datatype": "INT64",
                "data": encoded["token_type_ids"].tolist()
            }
        ],
        "outputs": [
            {"name": "last_hidden_state"}
        ]
    }

    start = time.perf_counter()
    resp = requests.post(f"{TRITON_URL}/v2/models/{MODEL_NAME}/infer",
                         json=payload)
    elapsed = time.perf_counter() - start

    if resp.status_code != 200:
        print(f"  ERROR: {resp.status_code} — {resp.text[:500]}")
        return None

    result = resp.json()
    output = result["outputs"][0]
    shape = output["shape"]
    data = output["data"]

    # Extract CLS token embedding (first token)
    embedding_dim = shape[-1]
    cls_embedding = data[:embedding_dim]

    # Normalize (sentence-transformers uses L2 normalization)
    norm = np.linalg.norm(cls_embedding)
    cls_normalized = [x / norm for x in cls_embedding]

    print(f"  Input text:    \"{text}\"")
    print(f"  Output shape:  {shape}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  CLS norm:      {norm:.4f}")
    print(f"  First 5 dims:  {[f'{x:.4f}' for x in cls_normalized[:5]]}")
    print(f"  Latency:       {elapsed*1000:.1f} ms")
    print()

    return cls_normalized


# ── Part C: Similarity Test ────────────────────────────────────────

def similarity_test(tokenizer):
    """Test that embeddings capture semantic similarity."""
    print("=" * 80)
    print("PART C: SEMANTIC SIMILARITY VALIDATION")
    print("=" * 80)
    print()

    sentences = [
        "GPU memory bandwidth limits inference speed",
        "The throughput of neural network inference depends on VRAM bandwidth",
        "I enjoy cooking pasta on weekends",
        "CUDA cores process tensor operations in parallel",
    ]

    embeddings = []
    for sent in sentences:
        encoded = tokenizer(sent, padding="max_length", max_length=128,
                            truncation=True, return_tensors="np")
        payload = {
            "inputs": [
                {"name": "input_ids", "shape": [1, 128], "datatype": "INT64",
                 "data": encoded["input_ids"].tolist()},
                {"name": "attention_mask", "shape": [1, 128], "datatype": "INT64",
                 "data": encoded["attention_mask"].tolist()},
                {"name": "token_type_ids", "shape": [1, 128], "datatype": "INT64",
                 "data": encoded["token_type_ids"].tolist()},
            ],
            "outputs": [{"name": "last_hidden_state"}]
        }
        resp = requests.post(f"{TRITON_URL}/v2/models/{MODEL_NAME}/infer",
                             json=payload)
        result = resp.json()
        data = result["outputs"][0]["data"]
        dim = result["outputs"][0]["shape"][-1]
        emb = np.array(data[:dim])
        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb)

    # Compute cosine similarity matrix
    print("  Cosine Similarity Matrix:")
    print(f"  {'':>4}", end="")
    for i in range(len(sentences)):
        print(f"  S{i+1:>2}", end="")
    print()

    for i in range(len(sentences)):
        print(f"  S{i+1:>2}", end="")
        for j in range(len(sentences)):
            sim = np.dot(embeddings[i], embeddings[j])
            print(f"  {sim:.2f}", end="")
        print(f"  \"{sentences[i][:50]}...\"" if len(sentences[i]) > 50
              else f"  \"{sentences[i]}\"")

    # Validate: S1-S2 should be high, S1-S3 should be low
    sim_12 = np.dot(embeddings[0], embeddings[1])
    sim_13 = np.dot(embeddings[0], embeddings[2])
    sim_14 = np.dot(embeddings[0], embeddings[3])

    print()
    print(f"  GPU bandwidth ↔ VRAM throughput:  {sim_12:.3f} (expect HIGH)")
    print(f"  GPU bandwidth ↔ cooking pasta:    {sim_13:.3f} (expect LOW)")
    print(f"  GPU bandwidth ↔ CUDA tensors:     {sim_14:.3f} (expect MEDIUM)")
    print(f"  Semantic coherence:               {'PASS ✓' if sim_12 > sim_13 + 0.2 else 'FAIL ✗'}")
    print()


# ── Part D: Dynamic Batching Benchmark ─────────────────────────────

async def send_triton_request(session, tokenizer, text, request_id):
    """Send a single async inference request."""
    encoded = tokenizer(text, padding="max_length", max_length=128,
                        truncation=True, return_tensors="np")
    payload = {
        "inputs": [
            {"name": "input_ids", "shape": [1, 128], "datatype": "INT64",
             "data": encoded["input_ids"].tolist()},
            {"name": "attention_mask", "shape": [1, 128], "datatype": "INT64",
             "data": encoded["attention_mask"].tolist()},
            {"name": "token_type_ids", "shape": [1, 128], "datatype": "INT64",
             "data": encoded["token_type_ids"].tolist()},
        ],
        "outputs": [{"name": "last_hidden_state"}]
    }

    start = time.perf_counter()
    try:
        async with session.post(f"{TRITON_URL}/v2/models/{MODEL_NAME}/infer",
                                json=payload,
                                timeout=aiohttp.ClientTimeout(total=30)) as resp:
            elapsed = time.perf_counter() - start
            if resp.status != 200:
                return {"success": False, "latency": elapsed, "error": f"HTTP {resp.status}"}
            await resp.json()
            return {"success": True, "latency": elapsed, "request_id": request_id}
    except Exception as e:
        return {"success": False, "latency": time.perf_counter() - start,
                "error": str(e)[:200]}


async def benchmark_dynamic_batching(tokenizer):
    """Benchmark Triton's dynamic batching at various concurrency levels."""
    print("=" * 80)
    print("PART D: DYNAMIC BATCHING THROUGHPUT BENCHMARK")
    print("=" * 80)
    print()

    texts = [
        "GPU memory bandwidth limits inference speed",
        "Neural networks require large amounts of training data",
        "Transformer models use self-attention mechanisms",
        "Quantization reduces model size with minimal accuracy loss",
        "Tensor parallelism splits layers across multiple GPUs",
        "CUDA graphs eliminate kernel launch overhead",
        "PagedAttention manages KV cache like virtual memory",
        "Dynamic batching accumulates requests for efficiency",
    ]

    concurrency_levels = [1, 2, 4, 8, 16, 32, 64, 128]
    num_iterations = 3
    results = {}

    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    async with aiohttp.ClientSession(connector=connector) as session:

        # Warmup
        print("  Warming up...")
        for i in range(5):
            await send_triton_request(session, tokenizer, texts[0], "warmup")
        print()

        for conc in concurrency_levels:
            iter_results = []
            for iteration in range(num_iterations):
                tasks = [
                    send_triton_request(session, tokenizer,
                                        texts[i % len(texts)], i)
                    for i in range(conc)
                ]
                wall_start = time.perf_counter()
                batch_results = await asyncio.gather(*tasks)
                wall_elapsed = time.perf_counter() - wall_start

                successful = [r for r in batch_results if r["success"]]
                if not successful:
                    continue

                latencies = [r["latency"] for r in successful]
                throughput = len(successful) / wall_elapsed

                iter_results.append({
                    "throughput": throughput,
                    "latency_mean": statistics.mean(latencies),
                    "latency_p95": sorted(latencies)[int(0.95 * len(latencies))]
                                   if len(latencies) > 1 else latencies[0],
                    "wall_time": wall_elapsed,
                    "successful": len(successful),
                })

            if iter_results:
                avg_tp = statistics.mean(r["throughput"] for r in iter_results)
                avg_lat = statistics.mean(r["latency_mean"] for r in iter_results)
                avg_p95 = statistics.mean(r["latency_p95"] for r in iter_results)

                results[conc] = {
                    "throughput": avg_tp,
                    "latency_mean": avg_lat,
                    "latency_p95": avg_p95,
                }

                print(f"  Concurrency {conc:>4}: "
                      f"{avg_tp:>8.1f} req/s | "
                      f"mean {avg_lat*1000:>7.1f}ms | "
                      f"p95 {avg_p95*1000:>7.1f}ms")

    # Summary
    print()
    if results:
        peak_conc = max(results, key=lambda c: results[c]["throughput"])
        peak_tp = results[peak_conc]["throughput"]
        single_tp = results.get(1, {}).get("throughput", 0)

        print(f"  Peak throughput:     {peak_tp:.1f} req/s at concurrency={peak_conc}")
        if single_tp:
            print(f"  Batching speedup:    {peak_tp/single_tp:.1f}x over single request")
        print(f"  Single-req latency:  {results.get(1, {}).get('latency_mean', 0)*1000:.1f}ms")
    print()


# ── Part E: Prometheus Metrics ─────────────────────────────────────

def check_metrics():
    """Fetch and display Triton Prometheus metrics."""
    print("=" * 80)
    print("PART E: TRITON PROMETHEUS METRICS")
    print("=" * 80)
    print()

    resp = requests.get("http://localhost:8003/metrics")
    lines = resp.text.split("\n")

    # Extract key metrics
    interesting = [
        "nv_inference_request_success",
        "nv_inference_request_failure",
        "nv_inference_count",
        "nv_inference_exec_count",
        "nv_inference_queue_duration_us",
        "nv_inference_compute_infer_duration_us",
        "nv_gpu_utilization",
        "nv_gpu_memory_used_bytes",
    ]

    for metric_name in interesting:
        for line in lines:
            if line.startswith(metric_name) and not line.startswith("#"):
                print(f"  {line}")
                break

    print()
    print("  Full metrics endpoint: http://localhost:8003/metrics")
    print("  (Prometheus-compatible, can be scraped by Grafana)")
    print()


# ── Main ───────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("WEEK 6 EXPERIMENT 3: TRITON INFERENCE SERVER — EMBEDDING MODEL")
    print("=" * 80)
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print()

    # Part A: Health & metadata
    if not check_health():
        return

    # Part B: Single inference
    single_inference(tokenizer)

    # Part C: Semantic similarity
    similarity_test(tokenizer)

    # Part D: Dynamic batching benchmark
    asyncio.run(benchmark_dynamic_batching(tokenizer))

    # Part E: Prometheus metrics
    check_metrics()

    print("=" * 80)
    print("EXPERIMENT 3 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()