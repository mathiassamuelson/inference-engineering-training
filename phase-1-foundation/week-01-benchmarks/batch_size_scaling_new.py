import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import pandas as pd
import os

model_name = "meta-llama/Llama-3.2-3B-Instruct"

print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="cuda:0"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Test different batch sizes
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1200]
results = []

print("\n" + "="*80)
print("BATCH SIZE SCALING ANALYSIS: THROUGHPUT & MEMORY")
print("="*80)

for batch_size in batch_sizes:
    print(f"\n[Testing batch_size={batch_size}]")

    try:
        prompts = ["Explain GPU memory:" for _ in range(batch_size)]

        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats(0)

        # Warmup
        print("  Warming up...")
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda:0")
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.pad_token_id)

        # Reset again after warmup
        torch.cuda.reset_peak_memory_stats(0)

        # Benchmark with memory tracking
        print("  Benchmarking...")
        iterations = 5  # Using 5 iterations like original
        tokens_per_generation = 50

        torch.cuda.synchronize()
        start = time.time()

        for i in range(iterations):
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda:0")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=tokens_per_generation, pad_token_id=tokenizer.pad_token_id)
            print(f"    Iteration {i+1}/{iterations}", end='\r')

        torch.cuda.synchronize()
        elapsed = time.time() - start

        # Calculate throughput metrics
        total_tokens = batch_size * tokens_per_generation * iterations
        total_throughput = total_tokens / elapsed
        throughput_per_sample = total_throughput / batch_size
        avg_time_per_batch = elapsed / iterations

        # Get memory stats
        memory_allocated_gb = torch.cuda.memory_allocated(0) / 1e9
        peak_memory_gb = torch.cuda.max_memory_allocated(0) / 1e9
        memory_reserved_gb = torch.cuda.max_memory_reserved(0) / 1e9

        results.append({
            'batch_size': batch_size,
            'total_throughput': total_throughput,
            'throughput_per_sample': throughput_per_sample,
            'avg_batch_time': avg_time_per_batch,
            'memory_allocated_gb': memory_allocated_gb,
            'peak_memory_gb': peak_memory_gb,
            'memory_reserved_gb': memory_reserved_gb,
            'status': 'SUCCESS'
        })

        print(f"  ✓ Total throughput: {total_throughput:7.1f} tok/s")
        print(f"  ✓ Per-sample throughput: {throughput_per_sample:5.1f} tok/s")
        print(f"  ✓ Avg batch time: {avg_time_per_batch:.3f}s")
        print(f"  ✓ Peak memory: {peak_memory_gb:.2f} GB")

        torch.cuda.empty_cache()

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"  ✗ OUT OF MEMORY")
            results.append({
                'batch_size': batch_size,
                'total_throughput': None,
                'throughput_per_sample': None,
                'avg_batch_time': None,
                'memory_allocated_gb': None,
                'peak_memory_gb': None,
                'memory_reserved_gb': None,
                'status': 'OOM'
            })
            torch.cuda.empty_cache()
            break
        else:
            raise

# Save results
df = pd.DataFrame(results)
os.makedirs('results', exist_ok=True)
df.to_csv('results/batch_size_scaling.csv', index=False)

print("\n" + "="*80)
print("SUMMARY: THROUGHPUT & MEMORY RESULTS")
print("="*80)
print("\nBatch | Total Throughput | Per-Sample | Peak Memory | Status")
print("------|------------------|------------|-------------|--------")
for _, row in df.iterrows():
    if row['status'] == 'SUCCESS':
        print(f"{row['batch_size']:5.0f} | {row['total_throughput']:10,.1f} tok/s | {row['throughput_per_sample']:6.1f} tok/s | {row['peak_memory_gb']:7.2f} GB | {row['status']}")
    else:
        print(f"{row['batch_size']:5.0f} | {'N/A':>16} | {'N/A':>10} | {'N/A':>11} | {row['status']}")

# Analyze memory growth
successful = df[df['status'] == 'SUCCESS'].copy()
if len(successful) > 1:
    print("\n" + "="*80)
    print("MEMORY GROWTH ANALYSIS")
    print("="*80)

    # Calculate memory per sample
    baseline_memory = successful.iloc[0]['peak_memory_gb']  # Memory at batch_size=1
    successful['memory_per_sample_mb'] = ((successful['peak_memory_gb'] - baseline_memory) * 1024) / (successful['batch_size'] - 1)

    # For batch_size=1, we can't calculate delta, so handle separately
    memory_increments = []
    for i in range(len(successful)):
        if i == 0:
            # First row - this is baseline
            continue
        else:
            # Calculate based on increment from baseline
            batch = successful.iloc[i]['batch_size']
            peak = successful.iloc[i]['peak_memory_gb']
            increment_mb = ((peak - baseline_memory) * 1024) / (batch - 1)
            memory_increments.append(increment_mb)

    avg_memory_per_sample = sum(memory_increments) / len(memory_increments) if memory_increments else 0

    print(f"\nModel baseline (batch_size=1): {baseline_memory:.2f} GB")
    print(f"Average memory per additional sample: {avg_memory_per_sample:.2f} MB")
    print(f"\nMemory breakdown by batch size:")
    print("\nBatch | Peak Memory | Memory/Sample")
    print("------|-------------|---------------")
    for i, row in successful.iterrows():
        if row['batch_size'] == 1:
            print(f"{row['batch_size']:5.0f} | {row['peak_memory_gb']:7.2f} GB | (baseline)")
        else:
            mem_per_sample = ((row['peak_memory_gb'] - baseline_memory) * 1024) / (row['batch_size'] - 1)
            print(f"{row['batch_size']:5.0f} | {row['peak_memory_gb']:7.2f} GB | {mem_per_sample:7.2f} MB")

# Calculate GPU utilization trend
if len(successful) > 0:
    print("\n" + "="*80)
    print("EFFICIENCY ANALYSIS")
    print("="*80)

    max_throughput_per_sample = successful['throughput_per_sample'].max()
    successful['efficiency'] = (successful['throughput_per_sample'] / max_throughput_per_sample) * 100

    print("\nBatch | Throughput/Sample | Efficiency | GPU Memory Used")
    print("------|-------------------|------------|------------------")
    for _, row in successful.iterrows():
        efficiency = (row['throughput_per_sample'] / max_throughput_per_sample) * 100
        memory_pct = (row['peak_memory_gb'] / 24.0) * 100
        print(f"{row['batch_size']:5.0f} | {row['throughput_per_sample']:11.1f} tok/s | {efficiency:7.1f}%   | {memory_pct:7.1f}% ({row['peak_memory_gb']:.2f} GB)")

    print(f"\nNote: Efficiency = (per-sample throughput / max per-sample throughput)")
    print(f"      As batch size increases, per-sample throughput typically decreases")
    print(f"      but total throughput and GPU utilization increase")

print(f"\nResults saved to: results/batch_size_scaling.csv")
print("="*80)