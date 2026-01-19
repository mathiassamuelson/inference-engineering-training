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

batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
results = []

print("\n" + "="*70)
print("BATCH SIZE SCALING WITH PROPER MEMORY TRACKING")
print("="*70)

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
        iterations = 3

        torch.cuda.synchronize()
        start = time.time()

        for i in range(iterations):
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda:0")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.pad_token_id)
            print(f"    Iteration {i+1}/{iterations}", end='\r')

        torch.cuda.synchronize()
        elapsed = time.time() - start

        throughput = (batch_size * 50 * iterations) / elapsed

        # Get memory stats
        memory_allocated_gb = torch.cuda.memory_allocated(0) / 1e9
        memory_reserved_gb = torch.cuda.memory_reserved(0) / 1e9
        max_memory_allocated_gb = torch.cuda.max_memory_allocated(0) / 1e9  # PEAK during generation!
        max_memory_reserved_gb = torch.cuda.max_memory_reserved(0) / 1e9

        results.append({
            'batch_size': batch_size,
            'total_throughput': throughput,
            'tokens_per_sec_per_sample': throughput / batch_size,
            'memory_after_gen_gb': memory_allocated_gb,
            'peak_memory_gb': max_memory_allocated_gb,  # This is what matters!
            'memory_reserved_gb': max_memory_reserved_gb,
            'status': 'SUCCESS'
        })

        print(f"  ✓ Throughput: {throughput:7.1f} tok/s")
        print(f"  ✓ Memory after: {memory_allocated_gb:.2f} GB")
        print(f"  ✓ Peak memory: {max_memory_allocated_gb:.2f} GB")  # <-- The real number

        torch.cuda.empty_cache()

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"  ✗ OUT OF MEMORY")
            results.append({
                'batch_size': batch_size,
                'total_throughput': None,
                'tokens_per_sec_per_sample': None,
                'memory_after_gen_gb': None,
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
df.to_csv('results/batch_size_scaling_corrected.csv', index=False)

print("\n" + "="*70)
print("RESULTS WITH PEAK MEMORY USAGE")
print("="*70)
print(df[['batch_size', 'total_throughput', 'peak_memory_gb', 'status']].to_string(index=False))

# Analyze memory growth
successful = df[df['status'] == 'SUCCESS'].copy()
if len(successful) > 1:
    print("\n" + "="*70)
    print("MEMORY GROWTH ANALYSIS")
    print("="*70)

    successful['memory_per_sample_mb'] = (successful['peak_memory_gb'] - 6.0) * 1024 / successful['batch_size']

    print(successful[['batch_size', 'peak_memory_gb', 'memory_per_sample_mb']].to_string(index=False))

    print(f"\nModel baseline: ~6.0 GB")
    print(f"Average KV cache per sample: {successful['memory_per_sample_mb'].mean():.1f} MB")