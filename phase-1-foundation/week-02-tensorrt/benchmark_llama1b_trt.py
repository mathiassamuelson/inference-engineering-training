#!/usr/bin/env python3
# Filename: benchmark_llama1b_trt.py
# Location: phase-1-foundation/week-02-tensorrt/
# Purpose: Benchmark Llama 3.2 1B with TensorRT via Optimum

import torch
import time
import os
import gc
import warnings

warnings.filterwarnings("ignore")

print("=" * 70)
print("EXPERIMENT 2: LLAMA 3.2 1B TENSORRT BENCHMARK")
print("=" * 70)
print("\nUsing 1B model to verify TensorRT pipeline")
print("(3B model ONNX export requires >60GB RAM)")

model_name = "meta-llama/Llama-3.2-1B-Instruct"
onnx_path = "./results/llama1b_onnx_cli"

# Verify ONNX exists
if not os.path.exists(os.path.join(onnx_path, "model.onnx")):
    print(f"\n✗ ONNX model not found at {onnx_path}")
    print("  Run export_llama1b_onnx_cli.sh first")
    exit(1)

# Test parameters
NUM_WARMUP = 3
NUM_ITERATIONS = 10
MAX_NEW_TOKENS = 100
PROMPT = "Explain how GPU memory bandwidth affects inference performance:"

from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(model_name)

results = {}

#######################################################################
# PHASE 1: PyTorch FP16 Baseline
#######################################################################
print("\n" + "=" * 70)
print("PHASE 1: PyTorch FP16 Baseline")
print("=" * 70)

print("\n[1/3] Loading PyTorch model...")
pt_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="cuda:0"
)
pt_model.eval()

mem_pytorch = torch.cuda.memory_allocated(0) / 1e9
print(f"  ✓ Model loaded")
print(f"  ✓ GPU memory: {mem_pytorch:.2f} GB")

# Warmup
print("\n[2/3] Warming up...")
inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda:0")
for _ in range(NUM_WARMUP):
    with torch.no_grad():
        _ = pt_model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id
        )

# Benchmark
print("\n[3/3] Benchmarking PyTorch FP16...")
torch.cuda.synchronize()
start = time.time()

for i in range(NUM_ITERATIONS):
    with torch.no_grad():
        outputs = pt_model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id
        )
    print(f"  Iteration {i+1}/{NUM_ITERATIONS}", end="\r")

torch.cuda.synchronize()
elapsed = time.time() - start

tokens_generated = MAX_NEW_TOKENS * NUM_ITERATIONS
results["pytorch"] = {
    "throughput": tokens_generated / elapsed,
    "latency": (elapsed / NUM_ITERATIONS) * 1000,
    "memory": mem_pytorch,
}

print(f"\n  ✓ Throughput: {results['pytorch']['throughput']:.1f} tokens/sec")
print(f"  ✓ Latency: {results['pytorch']['latency']:.1f} ms per generation")

# Sample output
sample_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\n  Sample (first 150 chars): {sample_text[:150]}...")

# Cleanup
del pt_model
gc.collect()
torch.cuda.empty_cache()
time.sleep(2)

#######################################################################
# PHASE 2: ONNX + CUDA Provider
#######################################################################
print("\n" + "=" * 70)
print("PHASE 2: ONNX Runtime + CUDA Provider")
print("=" * 70)

try:
    print("\n[1/3] Loading ONNX model with CUDA provider...")

    ort_cuda = ORTModelForCausalLM.from_pretrained(
        onnx_path, provider="CUDAExecutionProvider"
    )

    mem_cuda = torch.cuda.memory_allocated(0) / 1e9
    print(f"  ✓ ONNX+CUDA model loaded")
    print(f"  ✓ GPU memory: {mem_cuda:.2f} GB")

    # Warmup
    print("\n[2/3] Warming up...")
    inputs = tokenizer(PROMPT, return_tensors="pt")
    for _ in range(NUM_WARMUP):
        _ = ort_cuda.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id
        )

    # Benchmark
    print("\n[3/3] Benchmarking ONNX + CUDA...")
    start = time.time()

    for i in range(NUM_ITERATIONS):
        outputs = ort_cuda.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id
        )
        print(f"  Iteration {i+1}/{NUM_ITERATIONS}", end="\r")

    elapsed = time.time() - start

    results["onnx_cuda"] = {
        "throughput": tokens_generated / elapsed,
        "latency": (elapsed / NUM_ITERATIONS) * 1000,
        "memory": mem_cuda,
    }

    print(f"\n  ✓ Throughput: {results['onnx_cuda']['throughput']:.1f} tokens/sec")
    print(f"  ✓ Latency: {results['onnx_cuda']['latency']:.1f} ms per generation")

    del ort_cuda
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

except Exception as e:
    print(f"\n  ✗ ONNX+CUDA failed: {type(e).__name__}")
    print(f"  Error: {str(e)[:300]}")
    results["onnx_cuda"] = None

#######################################################################
# PHASE 3: ONNX + TensorRT Provider
#######################################################################
print("\n" + "=" * 70)
print("PHASE 3: ONNX Runtime + TensorRT Provider")
print("=" * 70)

try:
    print("\n[1/3] Loading ONNX model with TensorRT provider...")
    print("  (First run builds TRT engines - may take several minutes)")

    ort_trt = ORTModelForCausalLM.from_pretrained(
        onnx_path, provider="TensorrtExecutionProvider"
    )

    mem_trt = torch.cuda.memory_allocated(0) / 1e9
    print(f"  ✓ ONNX+TRT model loaded")
    print(f"  ✓ GPU memory: {mem_trt:.2f} GB")

    # Warmup (TRT engine compilation happens here)
    print("\n[2/3] Warming up (TensorRT engine building)...")
    inputs = tokenizer(PROMPT, return_tensors="pt")
    for w in range(NUM_WARMUP):
        _ = ort_trt.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id
        )
        print(f"  Warmup {w+1}/{NUM_WARMUP}", end="\r")
    print()

    # Benchmark
    print("\n[3/3] Benchmarking ONNX + TensorRT...")
    start = time.time()

    for i in range(NUM_ITERATIONS):
        outputs = ort_trt.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id
        )
        print(f"  Iteration {i+1}/{NUM_ITERATIONS}", end="\r")

    elapsed = time.time() - start

    results["onnx_trt"] = {
        "throughput": tokens_generated / elapsed,
        "latency": (elapsed / NUM_ITERATIONS) * 1000,
        "memory": mem_trt,
    }

    print(f"\n  ✓ Throughput: {results['onnx_trt']['throughput']:.1f} tokens/sec")
    print(f"  ✓ Latency: {results['onnx_trt']['latency']:.1f} ms per generation")

except Exception as e:
    print(f"\n  ✗ ONNX+TRT failed: {type(e).__name__}")
    print(f"  Error: {str(e)[:300]}")
    results["onnx_trt"] = None

#######################################################################
# RESULTS SUMMARY
#######################################################################
print("\n" + "=" * 70)
print("RESULTS SUMMARY: LLAMA 3.2 1B")
print("=" * 70)

pytorch_tp = results["pytorch"]["throughput"]

print("\n┌─────────────────────┬───────────────┬─────────────┬──────────┬──────────┐")
print("│ Configuration       │ Throughput    │ Latency     │ Memory   │ Speedup  │")
print("├─────────────────────┼───────────────┼─────────────┼──────────┼──────────┤")

print(
    f"│ PyTorch FP16        │ {pytorch_tp:>8.1f} tok/s │ {results['pytorch']['latency']:>7.1f} ms  │ {results['pytorch']['memory']:>5.2f} GB │ 1.00x    │"
)

if results.get("onnx_cuda"):
    speedup = results["onnx_cuda"]["throughput"] / pytorch_tp
    print(
        f"│ ONNX + CUDA         │ {results['onnx_cuda']['throughput']:>8.1f} tok/s │ {results['onnx_cuda']['latency']:>7.1f} ms  │ {results['onnx_cuda']['memory']:>5.2f} GB │ {speedup:.2f}x    │"
    )
else:
    print(
        f"│ ONNX + CUDA         │ {'FAILED':>13} │ {'N/A':>11} │ {'N/A':>8} │ {'N/A':>8} │"
    )

if results.get("onnx_trt"):
    speedup = results["onnx_trt"]["throughput"] / pytorch_tp
    print(
        f"│ ONNX + TensorRT     │ {results['onnx_trt']['throughput']:>8.1f} tok/s │ {results['onnx_trt']['latency']:>7.1f} ms  │ {results['onnx_trt']['memory']:>5.2f} GB │ {speedup:.2f}x    │"
    )
else:
    print(
        f"│ ONNX + TensorRT     │ {'FAILED':>13} │ {'N/A':>11} │ {'N/A':>8} │ {'N/A':>8} │"
    )

print("└─────────────────────┴───────────────┴─────────────┴──────────┴──────────┘")

# Analysis
print("\n" + "-" * 70)
print("ANALYSIS")
print("-" * 70)

if results.get("onnx_cuda") and results.get("onnx_trt"):
    cuda_speedup = results["onnx_cuda"]["throughput"] / pytorch_tp
    trt_speedup = results["onnx_trt"]["throughput"] / pytorch_tp

    if trt_speedup > cuda_speedup:
        print(f"\n✓ TensorRT provides {trt_speedup:.2f}x speedup over PyTorch")
        print(
            f"  Additional {((trt_speedup/cuda_speedup)-1)*100:.1f}% gain over ONNX+CUDA"
        )
    else:
        print(
            f"\n⚠ TensorRT ({trt_speedup:.2f}x) not faster than CUDA ({cuda_speedup:.2f}x)"
        )
        print("  Possible reasons: small model, overhead dominance")

print("\n" + "=" * 70)
print("EXPERIMENT 2 COMPLETE")
print("=" * 70)
