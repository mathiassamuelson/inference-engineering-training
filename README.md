# AI Infrastructure Training: From Product Architect to AI Engineer or AI Product Manager

**Goal:** Master NVIDIA's AI stack through hands-on learning with a 4x RTX 3090 inference platform

**Background:** Principal Product Architect with 13+ years in carrier-scale infrastructure, transitioning to AI engineering/product management

**Hardware:** 4x NVIDIA RTX 3090 (96GB total VRAM) | Ubuntu 24.04 | CUDA 12.x

## 🎯 Training Objectives

1. **Technical Mastery:** NVIDIA inference stack (TensorRT, Triton, vLLM)
2. **Hands-on Experience:** Deploy and optimize real AI workloads
3. **Product Knowledge:** Cost modeling, latency trade-offs, production readiness
4. **Portfolio:** Working projects demonstrating end-to-end capability

## 📊 Current Progress

- **Phase 1:** Foundation (Weeks 1-4) - ✅ In Progress
- **Phase 2:** Production Inference (Weeks 5-8) - ⏳ Upcoming
- **Phase 3:** Optimization (Weeks 9-12) - ⏳ Upcoming
- **Phase 4:** Projects (Weeks 13-16) - ⏳ Upcoming
- **Phase 5:** Product Management (Weeks 17-20) - ⏳ Upcoming
- **Phase 6:** Capstone (Weeks 21-24) - ⏳ Upcoming

[Full Training Plan →](docs/training-plan.md)

## 🔬 Key Findings So Far

### Week 1: Baseline Benchmarks (Llama 3.2 3B)
- **FP16 vs FP32:** 1.56x speedup (memory bandwidth limited)
- **Throughput:** 84 tokens/sec on single RTX 3090
- **Learning:** Small models don't benefit from multi-GPU
- [Full Analysis →](phase-1-foundation/week-01-benchmarks/week-01.md)

## 📁 Repository Structure

- **`phase-N-name/`** - Organized by training phases
- **`docs/`** - Weekly reports, learnings, analysis
- **`tools/`** - Reusable utilities and helpers
- **`phase-4-projects/`** - Portfolio projects
- **`phase-6-capstone/`** - Final comprehensive project

## 🚀 Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/rtx3090-ai-training.git
cd rtx3090-ai-training

# Setup environment
./setup.sh

# Run Week 1 baseline benchmark
cd phase-1-foundation/week-01-benchmarks
python3 baseline_benchmark.py
```

## 📈 Hardware Progression

- **Weeks 1-2:** 2x RTX 3090 (48GB)
- **Week 3+:** 4x RTX 3090 (96GB) - enables 70B+ models

## 🎓 Background & Context

Coming from platform/infrastructure product management at Akamai (managing DNS security for 30+ carriers, 20M+ end users), I'm systematically building AI engineering skills to transition into AI-focused roles. This repository documents my journey and serves as a portfolio of hands-on learning.

## 📫 Connect

- LinkedIn: [Your Profile]
- Blog: [If you have one]
- Email: [Your Email]

## 📝 License

MIT License - Feel free to use any code or approaches documented here

---

**Last Updated:** January 2026
