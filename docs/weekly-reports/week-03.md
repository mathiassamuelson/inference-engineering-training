# Week 3: Multi-GPU Orchestration

**Duration:** February 2026  
**Hardware:** 4x RTX 3090 (24GB each), Ubuntu 24.04 LTS, CUDA 12.6  
**PCIe Topology:** GPU 0 on x16 (CPU-direct), GPUs 1-3 on x1 (chipset-routed)  
**Test Models:** Llama 3.2 3B Instruct, Llama 3.1 8B Instruct

---

## Executive Summary

Week 3 systematically evaluated multi-GPU strategies on a 4x RTX 3090 system with asymmetric PCIe topology. Four experiments produced five key findings:

1. **PCIe x1 has zero impact on inference throughput** — all 4 GPUs produce identical tok/s once weights are loaded, because inference runs entirely on-chip

2. **Data parallelism is the optimal strategy for PCIe systems** — 93.6% scaling efficiency at 4 GPUs with batch=32, delivering 7,422 tok/s total system throughput

3. **Pipeline parallelism costs 8-18% throughput** — overhead comes from synchronization latency and pipeline bubbles, not bandwidth, and worsens with larger batch sizes

4. **CUDA operations on different GPUs are naturally asynchronous** — 99.8% concurrent efficiency, explaining mechanically why data parallelism scales

5. **Multi-stream overlap only helps for different operation types** — compute-saturated GPUs can't overlap additional compute, but compute + data transfer overlap works perfectly

**Key takeaway:** On PCIe-connected systems without NVLink, data parallelism is the only viable multi-GPU inference strategy. Tensor parallelism is unusable, and pipeline parallelism should only be used when a model physically cannot fit on a single GPU.

---

## Objectives

- ✅ Map GPU topology and measure inter-GPU communication bandwidth
- ✅ Evaluate data parallelism scaling across 1-4 GPUs
- ✅ Measure pipeline parallelism overhead on PCIe x1 interconnects
- ✅ Understand CUDA streams, async execution, and synchronization costs
- ✅ Determine optimal multi-GPU strategy for this hardware

---

## Hardware Topology Discovery

### PCIe Lane Allocation

Investigation of the Gigabyte B650 Eagle AX motherboard revealed a critical hardware constraint:

| Slot | Electrical Wiring | PCIe Generation | Max Bandwidth | GPU |
|------|------------------|-----------------|---------------|-----|
| PCIEX16 | x16 (CPU-direct) | PCIe 4.0 | ~25 GB/s | GPU 0 |
| PCIEX1_1 | x1 (CPU) | PCIe 3.0 | ~1 GB/s | GPU 1 |
| PCIEX1_2 | x1 (CPU) | PCIe 3.0 | ~1 GB/s | GPU 2 |
| PCIEX1_3 | x1 (Chipset) | PCIe 3.0 | ~1 GB/s | GPU 3 |

Slots are physically x16 (cards fit) but electrically x1. This is by motherboard design, not a configuration issue. All GPUs report `nvidia-smi topo -m` as PHB (PCIe Host Bridge) connections with no peer-to-peer access.

### Measured Communication Bandwidth

| Path | Bandwidth | Latency (256 MB) |
|------|-----------|-------------------|
| GPU 0 ↔ GPUs 1,2,3 | 1.37-1.45 GB/s | 172-183 ms |
| GPU 1 ↔ GPU 2,3 | 0.75-0.77 GB/s | 325-332 ms |
| Host → GPU 0 | 18.44 GB/s | — |
| Host → GPUs 1,2,3 | 1.47-1.55 GB/s | — |
| Small transfer latency (4 KB) | — | 0.015-0.028 ms |

GPU 0's 18.44 GB/s host bandwidth confirms its PCIe 4.0 x16 connection. GPUs 1-3 at ~1.5 GB/s confirms x1 electrical wiring. Cross-GPU transfers between GPUs 1-3 at ~0.76 GB/s traverse two hops (chipset → CPU → chipset).

### Implications for Parallelism Strategies

Ring all-reduce measured at 378.9ms for 32 MB — making tensor parallelism (which requires all-reduce per layer) completely unviable. For an 80-layer model, this would add ~30 seconds of communication overhead per token.

---

## Experiment 1: GPU Topology & Communication Baseline

### Setup

Custom benchmark measuring GPU-to-GPU bandwidth (256 MB transfers), small transfer latency (4 KB), and simulated ring all-reduce (32 MB) across all GPU pairs.

### Key Results

- Average GPU-to-GPU bandwidth: 1.09 GB/s (vs. 12-13 GB/s expected on proper PCIe 3.0 x16)
- Ring all-reduce: 378.9ms for 32 MB (effective bandwidth: 0.06 GB/s)
- No peer-to-peer access between any GPU pair

### Finding

Hardware topology is the first thing to characterize in any multi-GPU system. The 10x bandwidth deficit vs. expectations completely changes which parallelism strategies are viable. Production infrastructure decisions require measuring actual interconnect performance, not assuming theoretical maximums.

---

## Experiment 2: Data Parallelism Throughput Scaling

### Setup

Llama 3.2 3B loaded as independent replicas on 1-4 GPUs. Each GPU runs inference independently with no inter-GPU communication. Tested at batch sizes 1, 8, and 32 per GPU.

### Results

**Individual GPU Performance (PCIe x1 vs x16):**

| GPU | PCIe Link | Throughput (batch=1) | vs. GPU 0 |
|-----|-----------|---------------------|-----------|
| GPU 0 | x16 | 83.9 tok/s | baseline |
| GPU 1 | x1 | 83.2 tok/s | 99.1% |
| GPU 2 | x1 | 84.2 tok/s | 100.4% |
| GPU 3 | x1 | 83.9 tok/s | 100.0% |

**Multi-GPU Scaling:**

| GPUs | Batch=1 Total | Efficiency | Batch=8 Total | Efficiency | Batch=32 Total | Efficiency |
|------|---------------|------------|---------------|------------|----------------|------------|
| 1x | 84 tok/s | 100% | 601 tok/s | 100% | 1,982 tok/s | 100% |
| 2x | 166 tok/s | 99.2% | 1,179 tok/s | 98.2% | 3,904 tok/s | 98.5% |
| 3x | 246 tok/s | 97.7% | 1,758 tok/s | 97.6% | 5,866 tok/s | 98.6% |
| 4x | 222 tok/s | 66.3% | 1,758 tok/s | 73.1% | 7,422 tok/s | 93.6% |

### Key Findings

**1. PCIe x1 Has Zero Impact on Inference**

All 4 GPUs produce identical throughput (83-84 tok/s). Once model weights are in VRAM, inference runs entirely on-chip. PCIe bandwidth only matters for model loading (one-time cost) and inter-GPU transfers (not needed for data parallelism).

**2. Near-Linear Scaling to 3 GPUs (97-99%)**

Data parallelism requires no inter-GPU communication during inference. Each GPU operates as an independent inference server. Scaling efficiency is near-perfect up to 3 GPUs across all batch sizes.

**3. 4th GPU Degrades at Low Batch Sizes (CPU Bottleneck)**

The 4-GPU scaling anomaly — 66% efficiency at batch=1 rising to 94% at batch=32 — traces to Python GIL contention. At batch=1, each GPU thread frequently returns to Python for token-by-token orchestration, and 4 threads competing for the GIL reduces each to ~66% of normal CPU time. At batch=32, GPU compute dominates and CPU overhead is hidden.

This was confirmed in Experiment 4D, where pure GPU compute (no Python orchestration) achieved 99.8% 4-GPU efficiency. The bottleneck is CPU-side, not GPU-side.

---

## Experiment 3: Pipeline Parallelism Performance

### Setup

Llama 3.1 8B (~16 GB in FP16) loaded on a single GPU (baseline), then forced into 2-GPU and 4-GPU pipeline configurations with explicit layer-to-GPU mapping. Tested at batch sizes 1 and 8.

### Results

**VRAM Distribution:**

| Configuration | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|---------------|-------|-------|-------|-------|
| 1 GPU (baseline) | 16.06 GB | — | — | — |
| 2-GPU pipeline | 8.04 GB | 8.03 GB | — | — |
| 4-GPU pipeline | 4.55 GB | 3.50 GB | 3.49 GB | 4.54 GB |

**Performance Comparison:**

| Configuration | Batch=1 Throughput | vs. Baseline | Batch=8 Throughput | vs. Baseline |
|---------------|-------------------|-------------|-------------------|-------------|
| 1 GPU (baseline) | 44.1 tok/s | 1.00x | 333.9 tok/s | 1.00x |
| 2-GPU pipeline | 40.8 tok/s | 0.92x | 279.7 tok/s | 0.84x |
| 4-GPU pipeline | 38.9 tok/s | 0.88x | 274.2 tok/s | 0.82x |

**Per-Stage-Boundary Overhead:**

| Batch Size | 2-GPU (1 boundary) | 4-GPU (3 boundaries) | Per Boundary |
|------------|--------------------|-----------------------|-------------|
| 1 | +1.86 ms/token | +3.01 ms/token | ~1.0 ms |
| 8 | +4.65 ms/token | +5.22 ms/token | ~1.7 ms |

### Key Findings

**1. Pipeline Overhead Is Synchronization, Not Bandwidth**

The per-boundary overhead of 1-2ms is far less than the ~30ms expected if transferring 32 MB at 1 GB/s. Per-token activation tensors are small: `hidden_dim × batch_size × 2 bytes`. At batch=1, that's only 8 KB for Llama 3.1 8B. The overhead is synchronization latency and pipeline bubble time, not data movement.

**2. Larger Batches Make Pipeline Parallelism Worse**

Throughput loss increases from 8-12% at batch=1 to 16-18% at batch=8. Activation tensor size scales linearly with batch size, increasing transfer time. This is the opposite of data parallelism, where larger batches improve efficiency.

**3. Pipeline Parallelism Is a Last Resort**

For any model that fits on a single GPU, pipeline splitting only hurts performance. It exists to enable running models that exceed single-GPU memory, accepting a throughput penalty as the cost of feasibility.

---

## Experiment 4: CUDA Streams & Async Execution

### Setup

Five sub-experiments probing CUDA's execution model: stream serialization, multi-stream overlap, compute+transfer overlap, multi-GPU independence, and synchronization cost.

### Results Summary

| Sub-Experiment | Key Measurement | Finding |
|----------------|----------------|---------|
| 4A: Default Stream | 1.93x for 2 workloads | Default stream serializes all operations |
| 4B: Multi-Stream (same GPU) | 11.6% overlap | Large matmuls saturate SMs, minimal overlap |
| 4C: Compute + Transfer | 401.3ms vs 401.5ms serial | Transfer fully hidden behind compute |
| 4D: Multi-GPU Concurrent | 99.8% efficiency | Different GPUs are naturally asynchronous |
| 4E: Sync Cost | -1.0% overhead | Workloads too long for sync to matter |

### Key Findings

**1. Streams Cannot Create More Compute**

Two 4096×4096 matmul streams on the same GPU achieved only 11.6% overlap. All 82 SMs were saturated by the first stream, leaving no capacity for the second. Streams enable overlap between different types of operations (compute + data movement), not doubling of compute throughput.

**2. Compute + Transfer Overlap Works Perfectly**

A 128 MB transfer (5.1ms) was completely hidden behind 396ms of compute. The GPU has a dedicated copy engine separate from the compute SMs, enabling true simultaneous execution. This is the foundation of production prefetching: while batch N processes, batch N+1's data transfers at zero effective cost.

**3. Pinned Memory Provides 40% Transfer Speedup**

Pinned (page-locked) memory: 5.1ms vs. regular memory: 7.2ms for 128 MB. Pinned memory enables DMA transfers without OS page management overhead. Production frameworks always use pinned memory for host-device transfers.

**4. Multi-GPU Operations Are Inherently Asynchronous**

Four GPUs running concurrently took 396.8ms — virtually identical to a single GPU at 395.9ms (99.8% efficiency). CUDA kernel launches are non-blocking from the CPU's perspective. This is the mechanical explanation for data parallelism's success: the CPU queues work on all GPUs before any GPU finishes, achieving perfect temporal overlap.

---

## Product & Engineering Insights

### 1. Hardware Topology Drives Architecture Decisions

The same 4-GPU system requires fundamentally different strategies depending on interconnect:

| Interconnect | Tensor Parallel | Pipeline Parallel | Data Parallel |
|-------------|----------------|-------------------|---------------|
| NVLink (600 GB/s) | ✅ Optimal | ✅ Good | ✅ Good |
| PCIe x16 (~25 GB/s) | ⚠️ Marginal | ✅ Viable | ✅ Optimal |
| PCIe x1 (~1 GB/s) | ❌ Unusable | ⚠️ Last resort | ✅ Optimal |

### 2. Capacity Planning: Data Parallel Configuration

**System throughput (4x RTX 3090, Llama 3.2 3B, data parallel):**

| Batch per GPU | Total Throughput | Concurrent Users (50 tok/s SLA) |
|---------------|-----------------|--------------------------------|
| 1 | 222 tok/s | 4 users |
| 8 | 1,758 tok/s | 35 users |
| 32 | 7,422 tok/s | 148 users |

**Compare to Week 1 single-GPU baseline:** Single GPU at batch=32 achieved ~1,982 tok/s. 4-GPU data parallel at batch=32 achieves 7,422 tok/s — a 3.74x improvement for 4x the hardware.

### 3. Cost Implications

For workloads that fit on a single GPU, adding GPUs via data parallelism provides near-linear throughput scaling:

| Configuration | Hardware Cost | Throughput (batch=32) | Cost per tok/s |
|---------------|--------------|----------------------|----------------|
| 1x RTX 3090 | ~$1,500 | 1,982 tok/s | $0.76 |
| 4x RTX 3090 (data parallel) | ~$6,000 | 7,422 tok/s | $0.81 |

Cost efficiency remains roughly constant — scaling GPUs via data parallelism is economically linear. No premium for multi-GPU; the slight efficiency loss (~6%) at 4 GPUs is the only cost.

### 4. When Pipeline Parallelism Is Justified

Pipeline parallelism should only be used when a model exceeds single-GPU memory. The break-even analysis:

| Model | FP16 Size | Fits on 1 GPU? | Strategy |
|-------|-----------|----------------|----------|
| Llama 3.2 3B | 6.4 GB | ✅ Yes | Data parallel (4 replicas) |
| Llama 3.1 8B | 16.1 GB | ✅ Yes | Data parallel (1 replica/GPU) |
| Llama 3.1 70B | ~140 GB | ❌ No | Pipeline across 4 GPUs (with quantization) |

---

## Technical Skills Developed

1. **PCIe topology analysis:** `nvidia-smi topo`, `lspci` for link negotiation, understanding LnkCap vs LnkSta
2. **Multi-GPU programming:** Device maps, explicit layer placement, thread-based data parallelism
3. **CUDA execution model:** Streams, async launches, synchronization patterns, pinned memory
4. **Performance analysis:** Isolating CPU vs GPU bottlenecks, identifying GIL contention
5. **Hardware-aware architecture:** Matching parallelism strategy to interconnect capability

---

## Challenges & Resolutions

### 1. Driver/Library Version Mismatch
**Problem:** `nvidia-smi` failed with "Driver/library version mismatch" after system update  
**Solution:** System reboot to load updated kernel module

### 2. PCIe x1 Bandwidth Discovery
**Problem:** Measured GPU-to-GPU bandwidth 10x lower than expected  
**Resolution:** Traced to motherboard electrical wiring (x1 slots), not a configuration issue. Confirmed via `lspci -vv` LnkSta showing Width x1.

### 3. 4-GPU Scaling Degradation at Low Batch
**Problem:** 4th GPU dropped efficiency to 66% at batch=1  
**Resolution:** Identified as Python GIL contention through isolation testing (Experiment 4D pure GPU compute showed 99.8% efficiency)

### 4. Hugging Face Gated Model Access
**Problem:** Llama 3.1 8B access restricted  
**Solution:** Accepted Meta license at huggingface.co/meta-llama. Proactively requested access to Llama 70B, Nemotron, Mistral, and Gemma for upcoming phases.

---

## Key Learnings: Theory vs Practice

### What I Expected
- PCIe x1 would degrade inference performance
- Pipeline parallelism overhead would be bandwidth-bound (~30ms per stage)
- Multi-stream would enable significant compute overlap
- 4-GPU data parallelism would scale ~4x

### What I Measured
- PCIe x1 has zero inference impact (99-100% of x16 performance)
- Pipeline overhead is synchronization-bound (~1-2ms per stage)
- Multi-stream achieves only 11.6% overlap on saturated GPUs
- 4-GPU data parallelism scales 3.74x (GIL limits at low batch)

### Critical Insight
**Hardware topology determines strategy, but not always in the ways you'd predict.** The PCIe x1 limitation was severe for inter-GPU communication but irrelevant for independent inference. Understanding which operations depend on interconnect bandwidth vs. which run entirely on-chip is essential for making correct architecture decisions.

---

## Interview Articulations

### Multi-GPU Strategy Selection
"On our 4x RTX 3090 system with PCIe x1 interconnects, we systematically evaluated all three parallelism strategies. Tensor parallelism was completely unviable — ring all-reduce took 379ms for 32 MB, which would add 30 seconds per token on an 80-layer model. Pipeline parallelism worked but cost 8-18% throughput due to synchronization latency and pipeline bubbles. Data parallelism achieved 93.6% scaling efficiency at batch=32 with zero inter-GPU communication. The key insight: hardware topology must be characterized before choosing a strategy — the same GPUs require completely different approaches depending on whether they're connected via NVLink, PCIe x16, or PCIe x1."

### PCIe Bandwidth and Inference
"We confirmed that PCIe x1 bandwidth has zero impact on inference throughput — all 4 GPUs produced identical 83-84 tok/s. This is because autoregressive inference runs entirely on-chip once weights are loaded into VRAM. PCIe bandwidth only matters for model loading (one-time startup cost) and inter-GPU communication (irrelevant for data parallelism). This finding has direct implications for infrastructure cost: consumer motherboards with limited PCIe lanes are perfectly viable for multi-GPU inference deployments using data parallelism."

### Pipeline Parallelism Overhead
"Pipeline parallelism on our system added 1-5ms overhead per stage boundary — surprisingly moderate because per-token activation tensors are small (8-64KB, not the full hidden state). The real cost is pipeline bubbles: during autoregressive generation, only one GPU is active at a time while others wait. We measured 8-18% throughput loss versus single-GPU, with larger batches making it worse because activation size scales with batch. Pipeline parallelism is a last resort for models that can't fit on one GPU."

### CUDA Async Execution
"CUDA operations on different GPUs are inherently asynchronous — we measured 99.8% concurrent efficiency across 4 GPUs. Kernel launches are non-blocking from the CPU's perspective, so the CPU queues work on all GPUs before any finishes. On the same GPU, streams enable overlapping different operation types (compute + data transfer) but can't overlap compute with compute when SMs are saturated. This is why production frameworks focus on overlapping data prefetching with inference computation, not on running multiple inference streams."

---

## Files Created

**Scripts:**
- `gpu_topology_benchmark.py` — PCIe topology and inter-GPU bandwidth measurement
- `data_parallel_scaling.py` — Multi-GPU data parallelism throughput analysis
- `pipeline_parallel_benchmark.py` — Pipeline parallelism overhead measurement
- `cuda_streams_benchmark.py` — CUDA async execution model experiments

**Documentation:**
- `week-03.md` (this report)

---

## Next Steps: Week 4 Preview

### With NVLink Bridge (if installed)
- Re-run topology benchmark to measure NVLink bandwidth vs PCIe
- Tensor parallelism experiments between GPU 0+1 NVLink pair
- Compare: NVLink tensor parallel vs. single GPU vs. data parallel
- Load Nemotron 70B across NVLink pair + remaining GPUs

### Without NVLink
- GPU memory profiling and management techniques
- CUDA memory allocator behavior and fragmentation
- Practical multi-GPU deployment patterns for production

### Preparation for Phase 2 (Weeks 5-8)
- Install vLLM and validate with Llama 3.2 3B
- Design comparison experiments: vLLM vs. transformers baseline from Week 1
- Plan Triton Inference Server deployment

---

## Conclusion

Week 3 established that **data parallelism is the dominant strategy for PCIe-connected consumer GPU systems.** While the hardware topology imposed severe limitations on inter-GPU communication (1 GB/s effective bandwidth), this proved irrelevant for the most effective parallelism strategy. Each GPU operates as an independent inference server, achieving 93.6% scaling efficiency at practical batch sizes.

The pipeline parallelism experiments provided essential context: splitting models across GPUs incurs real overhead (8-18% throughput loss), confirming that model splitting should only happen when models exceed single-GPU memory capacity.

The CUDA streams experiments built foundational understanding of why these patterns work: GPU operations across devices are inherently asynchronous, compute and data transfer can overlap via separate engine paths, and synchronization overhead matters most for rapid small-kernel workloads.

Combined with Week 1's discovery that the transformers library's batch scaling plateaus at ~5,000 tok/s, the path forward is clear: Phase 2's production frameworks (vLLM, Triton) must solve both the batch scaling and multi-GPU orchestration challenges simultaneously.

**Week 3 Status:** ✅ Complete — All objectives met

**Ready for Week 4:** GPU memory management and advanced multi-GPU patterns

---

*Report generated: February 2026*  
*Hardware: 4x RTX 3090, Gigabyte B650 Eagle AX, Ubuntu 24.04, CUDA 12.6*  
*PCIe topology: 1x x16 + 3x x1*