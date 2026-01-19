import torch
import time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_name = "meta-llama/Llama-3.2-3B-Instruct"
prompt = "Explain how GPU memory bandwidth affects inference performance:"
iterations = 10

results = []

print("="*60)
print("LLAMA 3.2 3B BASELINE BENCHMARK")
print("="*60)

# Test 1: Single GPU, FP32
print("\n[1/3] Testing Single GPU FP32...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,  # Fixed: using dtype instead of torch_dtype
    device_map="cuda:0"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Warmup
print("  Warming up...")
for _ in range(3):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    _ = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)

# Benchmark
print("  Benchmarking...")
torch.cuda.synchronize()
start = time.time()
for i in range(iterations):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    print(f"    Iteration {i+1}/{iterations}", end='\r')
torch.cuda.synchronize()
elapsed = time.time() - start

results.append({
    'config': 'Single GPU FP32',
    'avg_time': elapsed/iterations,
    'tokens_per_sec': (100*iterations)/elapsed,
    'memory_allocated': torch.cuda.memory_allocated(0)/1e9
})

print(f"\n  ✓ Throughput: {(100*iterations)/elapsed:.2f} tokens/sec")
print(f"  ✓ Memory: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

# Clear memory
del model
torch.cuda.empty_cache()
time.sleep(2)

# Test 2: Single GPU, FP16
print("\n[2/3] Testing Single GPU FP16...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="cuda:0"
)

# Warmup
print("  Warming up...")
for _ in range(3):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    _ = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)

# Benchmark
print("  Benchmarking...")
torch.cuda.synchronize()
start = time.time()
for i in range(iterations):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    print(f"    Iteration {i+1}/{iterations}", end='\r')
torch.cuda.synchronize()
elapsed = time.time() - start

results.append({
    'config': 'Single GPU FP16',
    'avg_time': elapsed/iterations,
    'tokens_per_sec': (100*iterations)/elapsed,
    'memory_allocated': torch.cuda.memory_allocated(0)/1e9
})

print(f"\n  ✓ Throughput: {(100*iterations)/elapsed:.2f} tokens/sec")
print(f"  ✓ Memory: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

# Clear memory
del model
torch.cuda.empty_cache()
time.sleep(2)

# Test 3: Dual GPU (device_map="auto")
print("\n[3/3] Testing Dual GPU (auto split)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto"
)

# Check how model was split
print("  Model distribution:")
for name, param in model.named_parameters():
    if param.device.type == 'cuda':
        print(f"    Layer on GPU {param.device.index}", end='\r')
print(f"\n    GPU 0: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
print(f"    GPU 1: {torch.cuda.memory_allocated(1)/1e9:.2f} GB")

# Warmup
print("  Warming up...")
for _ in range(3):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    _ = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)

# Benchmark
print("  Benchmarking...")
torch.cuda.synchronize()
start = time.time()
for i in range(iterations):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    print(f"    Iteration {i+1}/{iterations}", end='\r')
torch.cuda.synchronize()
elapsed = time.time() - start

results.append({
    'config': 'Dual GPU (auto)',
    'avg_time': elapsed/iterations,
    'tokens_per_sec': (100*iterations)/elapsed,
    'memory_allocated': (torch.cuda.memory_allocated(0) + torch.cuda.memory_allocated(1))/1e9
})

print(f"\n  ✓ Throughput: {(100*iterations)/elapsed:.2f} tokens/sec")
print(f"  ✓ Total Memory: {(torch.cuda.memory_allocated(0) + torch.cuda.memory_allocated(1))/1e9:.2f} GB")

# Save results
df = pd.DataFrame(results)
os.makedirs('results', exist_ok=True)
df.to_csv('results/week1_baseline.csv', index=False)

# Print summary
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(df.to_string(index=False))

# Calculate speedups
fp32_speed = results[0]['tokens_per_sec']
fp16_speed = results[1]['tokens_per_sec']
dual_speed = results[2]['tokens_per_sec']

print(f"\n" + "="*60)
print("SPEEDUP ANALYSIS")
print("="*60)
print(f"FP16 vs FP32:            {fp16_speed/fp32_speed:.2f}x faster")
print(f"Dual GPU vs Single FP16: {dual_speed/fp16_speed:.2f}x")

if dual_speed < fp16_speed:
    print("\nNote: Dual GPU is SLOWER - this is expected for small models")
    print("Reason: Communication overhead > computation savings for 3B params")

print(f"\nResults saved to: results/week1_baseline.csv")
