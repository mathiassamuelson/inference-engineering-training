#!/usr/bin/env python3
# Filename: benchmark_llama_optimum_trt.py
# Location: phase-1-foundation/week-02-tensorrt/
# Purpose: Benchmark Llama 3.2 3B with TensorRT via Optimum
# Compares against Week 1 baseline (84 tok/s PyTorch FP16)

import torch
import time
import os
import warnings

warnings.filterwarnings("ignore")

print("=" * 70)
print("EXPERIMENT 2: LLAMA 3.2 3B TENSORRT BENCHMARK")
print("=" * 70)

model_name = "meta-llama/Llama-3.2-3B-Instruct"
output_dir = "./results/llama_onnx_optimum"
os.makedirs(output_dir, exist_ok=True)

# Test parameters
NUM_WARMUP = 3
NUM_ITERATIONS = 10
MAX_NEW_TOKENS = 100
PROMPT = "Explain how GPU memory bandwidth affects inference performance:"

#######################################################################
# PHASE 1: PyTorch Baseline (reproduce Week 1)
#######################################################################
print("\n" + "=" * 70)
print("PHASE 1: PyTorch FP16 Baseline")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

print("\n[1/3] Loading PyTorch model...")
pt_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="cuda:0"
)
pt_tokenizer = AutoTokenizer.from_pretrained(model_name)
pt_model.eval()

mem_pytorch = torch.cuda.memory_allocated(0) / 1e9
print(f"  ✓ Model loaded")
print(f"  ✓ GPU memory: {mem_pytorch:.2f} GB")

# Warmup
print("\n[2/3] Warming up...")
inputs = pt_tokenizer(PROMPT, return_tensors="pt").to("cuda:0")
for _ in range(NUM_WARMUP):
    with torch.no_grad():
        _ = pt_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=pt_tokenizer.eos_token_id,
        )

# Benchmark
print("\n[3/3] Benchmarking PyTorch FP16...")
torch.cuda.synchronize()
start = time.time()

for i in range(NUM_ITERATIONS):
    with torch.no_grad():
        outputs = pt_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=pt_tokenizer.eos_token_id,
        )
    print(f"  Iteration {i+1}/{NUM_ITERATIONS}", end="\r")

torch.cuda.synchronize()
elapsed_pytorch = time.time() - start

tokens_generated = MAX_NEW_TOKENS * NUM_ITERATIONS
pytorch_throughput = tokens_generated / elapsed_pytorch
pytorch_latency = (elapsed_pytorch / NUM_ITERATIONS) * 1000  # ms

print(f"\n  ✓ Total time: {elapsed_pytorch:.2f}s")
print(f"  ✓ Throughput: {pytorch_throughput:.1f} tokens/sec")
print(f"  ✓ Latency: {pytorch_latency:.1f} ms per generation")

# Sample output
sample_text = pt_tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\n  Sample output (first 200 chars):")
print(f"  {sample_text[:200]}...")

# Free memory
del pt_model
torch.cuda.empty_cache()
time.sleep(2)

#######################################################################
# PHASE 2: ONNX Runtime with CUDA Provider
#######################################################################
print("\n" + "=" * 70)
print("PHASE 2: ONNX Runtime + CUDA Provider")
print("=" * 70)

from optimum.onnxruntime import ORTModelForCausalLM

print("\n[1/3] Exporting/Loading ONNX model (CUDA provider)...")
print("  (First run will export to ONNX - may take 5-10 minutes)")

try:
    # Try to load from cache first, otherwise export
    onnx_path_cuda = os.path.join(output_dir, "onnx_cuda")

    if os.path.exists(onnx_path_cuda):
        print("  Loading from cache...")
        ort_model_cuda = ORTModelForCausalLM.from_pretrained(
            onnx_path_cuda, provider="CUDAExecutionProvider"
        )
    else:
        print("  Exporting to ONNX (this takes a while)...")
        ort_model_cuda = ORTModelForCausalLM.from_pretrained(
            model_name, export=True, provider="CUDAExecutionProvider"
        )
        # Save for future runs
        ort_model_cuda.save_pretrained(onnx_path_cuda)
        print(f"  ✓ Saved to {onnx_path_cuda}")

    ort_tokenizer = AutoTokenizer.from_pretrained(model_name)

    mem_onnx = torch.cuda.memory_allocated(0) / 1e9
    print(f"  ✓ ONNX model loaded")
    print(f"  ✓ GPU memory: {mem_onnx:.2f} GB")

    # Warmup
    print("\n[2/3] Warming up...")
    inputs = ort_tokenizer(PROMPT, return_tensors="pt").to("cuda:0")
    for _ in range(NUM_WARMUP):
        _ = ort_model_cuda.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=ort_tokenizer.eos_token_id,
        )

    # Benchmark
    print("\n[3/3] Benchmarking ONNX + CUDA...")
    torch.cuda.synchronize()
    start = time.time()

    for i in range(NUM_ITERATIONS):
        outputs = ort_model_cuda.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=ort_tokenizer.eos_token_id,
        )
        print(f"  Iteration {i+1}/{NUM_ITERATIONS}", end="\r")

    torch.cuda.synchronize()
    elapsed_onnx_cuda = time.time() - start

    onnx_cuda_throughput = tokens_generated / elapsed_onnx_cuda
    onnx_cuda_latency = (elapsed_onnx_cuda / NUM_ITERATIONS) * 1000

    print(f"\n  ✓ Total time: {elapsed_onnx_cuda:.2f}s")
    print(f"  ✓ Throughput: {onnx_cuda_throughput:.1f} tokens/sec")
    print(f"  ✓ Latency: {onnx_cuda_latency:.1f} ms per generation")

    onnx_cuda_success = True

    # Free memory
    del ort_model_cuda
    torch.cuda.empty_cache()
    time.sleep(2)

except Exception as e:
    print(f"\n  ✗ ONNX CUDA failed: {type(e).__name__}")
    print(f"  Error: {str(e)[:500]}")
    onnx_cuda_success = False
    onnx_cuda_throughput = None
    onnx_cuda_latency = None

#######################################################################
# PHASE 3: ONNX Runtime with TensorRT Provider
#######################################################################
print("\n" + "=" * 70)
print("PHASE 3: ONNX Runtime + TensorRT Provider")
print("=" * 70)

print("\n[1/3] Loading ONNX model (TensorRT provider)...")
print("  (TensorRT will build optimized engines - may take 10-20 minutes first run)")

try:
    onnx_path_trt = os.path.join(output_dir, "onnx_trt")

    if os.path.exists(onnx_path_cuda):  # Reuse ONNX from CUDA export
        print("  Loading ONNX from cache, applying TensorRT...")
        ort_model_trt = ORTModelForCausalLM.from_pretrained(
            onnx_path_cuda, provider="TensorrtExecutionProvider"
        )
    else:
        print("  Exporting to ONNX with TensorRT...")
        ort_model_trt = ORTModelForCausalLM.from_pretrained(
            model_name, export=True, provider="TensorrtExecutionProvider"
        )

    mem_trt = torch.cuda.memory_allocated(0) / 1e9
    print(f"  ✓ TensorRT model loaded")
    print(f"  ✓ GPU memory: {mem_trt:.2f} GB")

    # Warmup (TensorRT builds engines during first inference)
    print("\n[2/3] Warming up (TensorRT engine building)...")
    inputs = ort_tokenizer(PROMPT, return_tensors="pt").to("cuda:0")
    for w in range(NUM_WARMUP):
        _ = ort_model_trt.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=ort_tokenizer.eos_token_id,
        )
        print(f"  Warmup {w+1}/{NUM_WARMUP}", end="\r")
    print()

    # Benchmark
    print("\n[3/3] Benchmarking ONNX + TensorRT...")
    torch.cuda.synchronize()
    start = time.time()

    for i in range(NUM_ITERATIONS):
        outputs = ort_model_trt.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=ort_tokenizer.eos_token_id,
        )
        print(f"  Iteration {i+1}/{NUM_ITERATIONS}", end="\r")

    torch.cuda.synchronize()
    elapsed_trt = time.time() - start

    trt_throughput = tokens_generated / elapsed_trt
    trt_latency = (elapsed_trt / NUM_ITERATIONS) * 1000

    print(f"\n  ✓ Total time: {elapsed_trt:.2f}s")
    print(f"  ✓ Throughput: {trt_throughput:.1f} tokens/sec")
    print(f"  ✓ Latency: {trt_latency:.1f} ms per generation")

    trt_success = True

except Exception as e:
    print(f"\n  ✗ TensorRT failed: {type(e).__name__}")
    print(f"  Error: {str(e)[:500]}")
    trt_success = False
    trt_throughput = None
    trt_latency = None

#######################################################################
# RESULTS SUMMARY
#######################################################################
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print("\n┌─────────────────────┬───────────────┬─────────────┬──────────┐")
print("│ Configuration       │ Throughput    │ Latency     │ Speedup  │")
print("├─────────────────────┼───────────────┼─────────────┼──────────┤")
print(
    f"│ PyTorch FP16        │ {pytorch_throughput:>8.1f} tok/s │ {pytorch_latency:>7.1f} ms  │ 1.00x    │"
)

if onnx_cuda_success:
    speedup_cuda = onnx_cuda_throughput / pytorch_throughput
    print(
        f"│ ONNX + CUDA         │ {onnx_cuda_throughput:>8.1f} tok/s │ {onnx_cuda_latency:>7.1f} ms  │ {speedup_cuda:.2f}x    │"
    )
else:
    print(f"│ ONNX + CUDA         │ {'FAILED':>13} │ {'N/A':>11} │ {'N/A':>8} │")

if trt_success:
    speedup_trt = trt_throughput / pytorch_throughput
    print(
        f"│ ONNX + TensorRT     │ {trt_throughput:>8.1f} tok/s │ {trt_latency:>7.1f} ms  │ {speedup_trt:.2f}x    │"
    )
else:
    print(f"│ ONNX + TensorRT     │ {'FAILED':>13} │ {'N/A':>11} │ {'N/A':>8} │")

print("└─────────────────────┴───────────────┴─────────────┴──────────┘")

print("\n" + "-" * 70)
print("Week 1 Reference: PyTorch FP16 baseline was 84 tok/s")
print("-" * 70)

if pytorch_throughput > 0:
    week1_comparison = pytorch_throughput / 84.0
    print(
        f"Today's PyTorch baseline: {pytorch_throughput:.1f} tok/s ({week1_comparison:.2f}x vs Week 1)"
    )

print("\n" + "=" * 70)
print("EXPERIMENT 2 COMPLETE")
print("=" * 70)
