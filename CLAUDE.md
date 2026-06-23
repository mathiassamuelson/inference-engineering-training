# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI infrastructure training repository for learning the NVIDIA inference stack (TensorRT, Triton, vLLM) using a 4x RTX 3090 platform. The project follows a 24-week structured learning plan progressing from fundamentals to production deployments.

## Commands

### Environment Setup
```bash
./setup.sh                    # Full environment setup (creates ai-inference venv)
source ai-inference/bin/activate  # Activate the virtual environment
```

### Running Benchmarks
```bash
cd phase-1-foundation/week-01-benchmarks
python3 baseline_benchmark.py  # Run Week 1 Llama 3.2 3B benchmark
```

### Development Tools
```bash
pytest                        # Run tests
black .                       # Format code
flake8                        # Lint code
```

## Architecture

### Directory Structure by Training Phase
- `phase-1-foundation/` - Weeks 1-4: NVIDIA stack fundamentals, multi-GPU orchestration
- `phase-2-production/` - Weeks 5-8: Triton server, vLLM deployment (upcoming)
- `phase-3-optimization/` - Weeks 9-12: Quantization, CUDA kernels (upcoming)
- `phase-4-projects/` - Weeks 13-16: Portfolio projects (RAG, video analytics, fine-tuning)
- `phase-5-product-mgmt/` - Weeks 17-20: Cost modeling, observability (upcoming)
- `phase-6-capstone/` - Weeks 21-24: Comprehensive project (upcoming)

### Key Components
- `tools/` - Reusable utilities and helper scripts
- `docs/training-plan.md` - Full 24-week curriculum
- `phase-N-*/week-NN*/` - weekly reports and journals, alongside their results and code

### Benchmark Scripts Pattern
Benchmark scripts (like `baseline_benchmark.py`) follow this pattern:
1. Load model with different configurations (FP32, FP16, multi-GPU)
2. Run warmup iterations
3. Benchmark with `torch.cuda.synchronize()` for accurate timing
4. Save results to CSV in `benchmarks/` or `results/` subdirectory

## Hardware Context

- 4x NVIDIA RTX 3090 (96GB total VRAM)
- Ubuntu 24.04 with CUDA 12.x
- Target workloads: LLM inference (Llama, Nemotron models)
- Weeks 1-2 use 2x GPUs; Week 3+ uses all 4 GPUs

## Key Dependencies

Core stack: PyTorch, Transformers, Accelerate, TensorRT, vLLM, Triton
Analysis: pandas, matplotlib, seaborn
Monitoring: prometheus-client, gpustat
