# Add to your benchmark script
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="cuda:0"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# FIX: Set pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token
print(f"Pad token set to: {tokenizer.pad_token}")

# Test different batch sizes
batch_sizes = [1, 2, 4, 8, 16, 32]
results = []

for batch_size in batch_sizes:
    # Create batch of prompts
    prompts = ["Explain GPU memory:" for _ in range(batch_size)]

    # Warmup
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda:0")
    _ = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(5):  # Fewer iterations for larger batches
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda:0")
        _ = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    throughput = (batch_size * 50 * 5) / elapsed
    memory = torch.cuda.memory_allocated(0) / 1e9

    results.append({
        'batch_size': batch_size,
        'throughput': throughput,
        'memory_gb': memory,
        'tokens_per_sec_per_sample': throughput / batch_size
    })

    print(
        f"Batch {batch_size:2d}: {throughput:6.1f} tok/s total, {throughput / batch_size:5.1f} tok/s/sample, {memory:.2f} GB")

import pandas as pd

df = pd.DataFrame(results)
df.to_csv('results/batch_size_scaling.csv', index=False)