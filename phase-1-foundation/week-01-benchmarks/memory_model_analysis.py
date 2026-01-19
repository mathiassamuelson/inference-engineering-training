import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Your actual measurements
data = {
    'batch_size': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1200],
    'peak_memory': [6.446801, 6.462780, 6.491525, 6.548607, 6.667163, 6.893360,
                     7.349540, 8.062959, 9.691371, 12.950030, 19.464464, 21.43]
}

df = pd.DataFrame(data)

# Fit to model: Memory = base + (kv_per_sample * batch_size)
def memory_model(batch_size, base, kv_per_sample):
    return base + (kv_per_sample * batch_size)

params, _ = curve_fit(memory_model, df['batch_size'], df['peak_memory'])
base_memory, kv_per_sample = params

print("="*70)
print("FINAL MEMORY MODEL FOR RTX 3090 + LLAMA 3.2 3B")
print("="*70)
print(f"\nPeak Memory = {base_memory:.3f} GB + {kv_per_sample*1000:.2f} MB × batch_size")
print(f"\nWhere:")
print(f"  Base = {base_memory:.3f} GB (model weights + fixed overhead)")
print(f"  KV cache = {kv_per_sample*1000:.2f} MB per sample")

# Predict max batch size
gpu_memory_limit = 24.0  # GB
max_batch_theoretical = int((gpu_memory_limit - base_memory) / kv_per_sample)

print(f"\n" + "="*70)
print("CAPACITY ANALYSIS")
print("="*70)
print(f"GPU Memory: {gpu_memory_limit} GB")
print(f"Theoretical max batch: {max_batch_theoretical}")
print(f"Practical max batch: ~1200 (allows 2.5 GB safety margin)")
print(f"Safety margin needed: ~{(1 - 21.43/24.0)*100:.1f}% (for fragmentation)")

# Test predictions
print(f"\n" + "="*70)
print("MODEL VALIDATION")
print("="*70)
print("Batch | Predicted | Actual | Error")
print("------|-----------|--------|-------")
for i, row in df.iterrows():
    predicted = memory_model(row['batch_size'], base_memory, kv_per_sample)
    error = abs(predicted - row['peak_memory']) / row['peak_memory'] * 100
    print(f"{int(row['batch_size']):5d} | {predicted:7.2f} GB | {row['peak_memory']:6.2f} GB | {error:5.2f}%")