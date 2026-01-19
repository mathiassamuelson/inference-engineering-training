# Week 1 Report

## Highlights
- Installed hardware.
- Installed Ubuntu 24.04 LTS.
- Installed NVIDIA drivers and CUDA toolkit.
- Installed PyTorch and required dependencies.
- Configured environment variables for CUDA and PyTorch.
- Verified PyTorch installation and CUDA compatibility.
- Using Claude AI, developed a training plan.
- Set up the Git repository and initial commit.
- Ran Baseline evaluation using Llama 3.2 3B model, see below.

## Baseline evaluation
- Model: Llama 3.2 3B

$ python3 baseline_benchmark.py
```
============================================================
RESULTS SUMMARY
============================================================
         config  avg_time  tokens_per_sec  memory_allocated
Single GPU FP32  1.852527       53.980307         12.859523
Single GPU FP16  1.190136       84.024019          6.434023
Dual GPU (auto)  1.189339       84.080317          6.434023

============================================================
SPEEDUP ANALYSIS
============================================================
FP16 vs FP32:            1.56x faster
Dual GPU vs Single FP16: 1.00x

Results saved to: benchmarks/week1_baseline.csv
```

- For a 3B parameter model in FP32, the memory usage is calculated by taking the number of parameters, 3B, and multiplying by 4 bytes (32 bits), which results in approximately 12.86 GB of memory usage.
- For a 3B parameter model in FP16, the memory usage is calculated by taking the number of parameters, 3B, and multiplying by 2 bytes (16 bits), which results in approximately 6.43 GB of memory usage.

### Analysis of results
1. FP16 Speedup: 1.56x (Lower than expected?)
2. Dual GPU: No Benefit (1.00x)
  * The model stayed on GPU 0:

    ```memory_allocated: 6.434023  # Only showing one GPU ```
  * With device_map="auto", Hugging Face saw the 3B model fits on one GPU and kept it there. No point splitting! 
    This is actually smart behavior - the framework avoided the PCIe tax.

3. The Memory Bandwidth
Let's calculate what we're actually achieving:

```FP16: 84 tokens/sec × 3B params × 2 bytes = 504 GB/sec```

**RTX 3090 spec: 936 GB/sec**

We're hitting 54% of peak bandwidth - that's actually quite good for real-world inference! The gap may come from:

- Cache inefficiencies
- Kernel launch overhead
- Non-perfect memory access patterns