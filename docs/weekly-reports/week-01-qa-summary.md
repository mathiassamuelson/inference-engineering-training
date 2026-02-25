# Week 1 Deep Dive: Technical Q&A Summary

This document summarizes key technical concepts explored during Week 1 analysis, focusing on inference bottlenecks, KV cache mechanics, and memory architecture relationships.

---

## Table of Contents

1. [Two Critical Bottlenecks](#two-critical-bottlenecks)
2. [GPU Memory Hierarchy](#gpu-memory-hierarchy)
3. [KV Cache Fundamentals](#kv-cache-fundamentals)
4. [Model Architecture Impact](#model-architecture-impact)
5. [Batch Size vs Sequence Length](#batch-size-vs-sequence-length)
6. [Context Window vs Context Length](#context-window-vs-context-length)
7. [What Gets Stored in KV Cache](#what-gets-stored-in-kv-cache)
8. [Understanding K and V Projections](#understanding-k-and-v-projections)
9. [Interview Articulations](#interview-articulations)

---

## Two Critical Bottlenecks

Week 1 revealed **two separate bottlenecks** that people often conflate:

### 1. Memory Bandwidth Bottleneck (Single-Request Performance)

**What it affects:** Why FP16 only gave 1.56x speedup instead of 2-3x

**Mechanism:**
- Each token generation requires loading 3B parameters (6 GB in FP16) from VRAM
- Model weights flow: VRAM → L2 Cache → L1 Cache → Tensor Cores
- Measured: 504 GB/s utilization (54% of RTX 3090's 936 GB/s peak)
- Tensor cores sit idle waiting for data

**Result:** 84 tok/s at batch=1, not 200+ tok/s

**Key insight:** Autoregressive decode is memory-bound, not compute-bound

### 2. Software Inefficiency Bottleneck (Batch Scaling)

**What it affects:** Why total throughput plateaus at ~5,000 tok/s

**Mechanism:**
- Python GIL overhead in generation loop
- Inefficient kernel launches for large batches
- No kernel fusion optimizations
- Sequential token generation poorly parallelized across batch dimension
- Framework not designed for production inference

**Result:** Per-sample throughput collapses from 84 → 4.2 tok/s (95% degradation) as batch increases

**Smoking gun:** If memory bandwidth were the issue, increasing batch size would help by amortizing weight loads. Instead, per-sample performance collapsed.

### The Critical Distinction

**If purely memory bandwidth limited:**
- Batch 1: 84 tok/s per sample
- Batch 1200: ~70-80 tok/s per sample (slight degradation)
- Total: ~84,000-96,000 tok/s

**Actual measurement:**
- Batch 1: 84 tok/s per sample
- Batch 1200: 4.2 tok/s per sample (95% collapse!)
- Total: ~5,000 tok/s (plateau)

The 95% per-sample degradation proves this is a **software parallelization problem**, not hardware bandwidth constraint.

---

## GPU Memory Hierarchy

### How Model Weights Flow

Model weights are **stored** in VRAM permanently but must be **read** for each computation:

```
VRAM (24 GB HBM2) - Permanent storage
    ↓ [936 GB/s bandwidth]
L2 Cache (6 MB) - Shared across GPU
    ↓
L1 Cache / Shared Memory (128 KB per SM) - Per-streaming multiprocessor
    ↓
Registers / Tensor Cores - Actual computation
```

### Autoregressive Generation Pattern

```
Token 1: Read 6 GB weights from VRAM → generate token 1
Token 2: Read 6 GB weights from VRAM → generate token 2 (KV cache updated)
Token 3: Read 6 GB weights from VRAM → generate token 3 (KV cache updated)
...
Token 50: Read 6 GB weights from VRAM → generate token 50
```

Each token generation requires a full forward pass through the model.

### Batching Advantage (Amortization)

**Batch = 1:**
```
Pass 1: Read 6 GB → compute 1 output
Pass 2: Read 6 GB → compute 1 output
... (50 passes for 50 tokens)
```

**Batch = 1200:**
```
Pass 1: Read 6 GB → compute 1200 outputs simultaneously
Pass 2: Read 6 GB → compute 1200 outputs simultaneously
... (50 passes for 50 tokens)
```

The weights are read the same number of times (50 passes), but with batching you compute 1200× more outputs per read. This is why larger batches **should** improve bandwidth utilization (but transformers library can't parallelize effectively).

---

## KV Cache Fundamentals

### The Problem Without KV Cache

Attention requires comparing every new token with all previous tokens:

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

**Naive approach (O(n²)):**
```
Token 1: Compute K₁, V₁ → Generate token 1
Token 2: Compute K₁, V₁, K₂, V₂ → Generate token 2 (recomputing K₁, V₁!)
Token 3: Compute K₁, V₁, K₂, V₂, K₃, V₃ → Generate token 3 (recomputing everything!)
...
```

Massively wasteful - recomputing K and V for old tokens every single time.

### The Solution: KV Cache (O(n))

**Store and reuse:**
```
Token 1:
  - Compute K₁, V₁
  - STORE in cache
  - Generate token 1

Token 2:
  - LOAD K₁, V₁ from cache
  - Compute K₂, V₂
  - APPEND to cache → [K₁, K₂], [V₁, V₂]
  - Generate token 2

Token 3:
  - LOAD K₁, K₂, V₁, V₂ from cache
  - Compute K₃, V₃
  - APPEND to cache → [K₁, K₂, K₃], [V₁, V₂, V₃]
  - Generate token 3
```

**Result:** Each step is O(1) for past tokens, only computing K, V for the new token.

### Why Only K and V?

In attention, you need three matrices per token:
- **Q (Query):** Only for the **current** token being generated (changes each time)
- **K (Key):** For **all** tokens (current + previous) - **reusable, CACHED**
- **V (Value):** For **all** tokens (current + previous) - **reusable, CACHED**

Q changes for every new token and can't be reused. K and V for previous tokens stay the same and can be cached.

### Memory Growth Pattern

KV cache grows linearly with sequence length:

**For Llama 3.2 3B:**
- Per-token theoretical: 0.33 MB
- Per-token measured: 0.26 MB (PyTorch optimizations)

**Your memory model:**
```
Peak Memory = 6.47 GB + 13.03 MB × batch_size
```

Where 13.03 MB = KV cache for 50 tokens per sample (0.26 MB/token × 50 tokens)

---

## Model Architecture Impact

### Memory Calculation from Architecture

**For Llama 3.2 3B:**
- 28 layers
- 24 attention heads
- 128 dimensions per head
- FP16 precision (2 bytes)

**Per-token KV cache size (Multi-Head Attention assumption):**
```
2 (K and V) × 28 layers × 24 heads × 128 dim × 2 bytes = 344,064 bytes ≈ 0.33 MB per token
```
**Correction (Week 4):** Llama 3.2 3B uses **Grouped Query Attention (GQA)**, not standard
Multi-Head Attention. GQA shares KV heads across multiple query heads — in this model,
24 query heads share 8 KV heads (3:1 ratio). Since only KV heads are cached, the actual
per-token cost is:
```
2 (K and V) × 28 layers × 8 KV heads × 128 dim × 2 bytes = 114,688 bytes ≈ 0.112 MB per token
```
This is 3x smaller than the MHA assumption. The original calculation above remains valid
as a reference for models that use standard MHA (e.g., GPT-2, older architectures), but
most modern LLMs use GQA specifically to reduce KV cache memory requirements.

**Verify with model config:**
```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
print(f"Query heads: {config.num_attention_heads}")      # 24
print(f"KV heads: {config.num_key_value_heads}")          # 8
```

### Breaking Down Each Component

**1. Why "2" (K and V)?**
- Each layer needs both Key and Value tensors for attention
- Query (Q) is computed fresh each time (not cached)

**2. Why "28 layers"?**
- Llama 3.2 3B has 28 transformer layers
- KV cache is **per-layer**, not shared
- Each layer transforms representations differently
- Memory implication: 28 layers = 28× the KV cache storage

**3. Why "24 heads"?**
- Multi-head attention splits into 24 parallel "heads"
- Each head learns different patterns (syntax, semantics, relationships)
- Total hidden dimension: 3,072 = 24 heads × 128 dim/head
- Memory implication: Must store K and V for each head separately

**4. Why "128 dim"?**
- Dimensionality per attention head
- For each head, K and V have shape: [sequence_length, 128]
- Design choice balancing capacity vs efficiency

**5. Why "2 bytes" (FP16)?**
- Standard precision for inference
- 2× memory reduction vs FP32
- Supported by Tensor Cores
- Minimal quality degradation

### Architecture Comparison

| Model | Layers | Heads | Head Dim | Hidden Dim | KV/Token (FP16) |
|-------|--------|-------|----------|------------|-----------------|
| Llama 3.2 1B | 16 | 32 | 64 | 2,048 | 131 KB |
| **Llama 3.2 3B** | **28** | **24** | **128** | **3,072** | **344 KB** |
| Llama 3.1 8B | 32 | 32 | 128 | 4,096 | 524 KB |
| Llama 3.1 70B | 80 | 64 | 128 | 8,192 | 2,621 KB |

**Key insight:** Larger models with more layers have proportionally larger KV cache requirements.

---

## Batch Size vs Sequence Length

### They Are Independent Dimensions

Batch size and sequence length are **orthogonal** - they don't "relate" to each other directly:

```
┌─────────────────────────────────────┐
│  Batch Size (how many requests)     │
│                                     │
│  Request 1: [tok₁, tok₂, ..., tok₅₀]  ← Sequence length
│  Request 2: [tok₁, tok₂, ..., tok₅₀]  ← Sequence length
│  Request 3: [tok₁, tok₂, ..., tok₅₀]  ← Sequence length
│  ...                                │
│  Request 1200: [tok₁, ..., tok₅₀]    ← Sequence length
└─────────────────────────────────────┘
```

### Definitions

**Sequence Length (context length):**
- How many tokens in a **single** request
- Example: Generate 500 tokens → sequence length = 500
- This is **per request**, independent of batch size

**Batch Size:**
- How many **independent requests** processing simultaneously
- Example: 64 users asking different questions → batch size = 64
- Each request can have different sequence lengths

### Full Memory Equation

```
Peak Memory = Base + (Per_Sample_Memory × Batch_Size)

Where:
Per_Sample_Memory = KV_Cache_Per_Token × Sequence_Length
Per_Sample_Memory = 0.2606 MB/token × Sequence_Length

Full equation:
Peak Memory = 6.47 GB + (0.2606 MB × Sequence_Length) × Batch_Size
```

### Example Scenarios

**Scenario 1: Short context, high batch**
- 1,200 requests (batch size = 1,200)
- 50 tokens each (sequence length = 50)
- Memory = 6.47 + (0.2606 × 50) × 1,200 = 21.7 GB

**Scenario 2: Long context, low batch**
- 28 requests (batch size = 28)
- 2,048 tokens each (sequence length = 2,048)
- Memory = 6.47 + (0.2606 × 2,048) × 28 = 21.4 GB

**Same GPU memory used, completely different workloads!**

### Production Tradeoff Matrix

| Workload Type | Batch Size | Seq Length | Use Case | Memory per GPU |
|---------------|-----------|------------|----------|----------------|
| Real-time chat | 1-8 | 50-200 | Low latency | Low (< 10 GB) |
| API serving | 64-128 | 100-500 | Balanced | Medium (10-20 GB) |
| Long document | 10-30 | 2048-4096 | Analysis | High (18-22 GB) |
| Batch processing | 256-512 | 50-100 | Max throughput | High (18-22 GB) |

---

## Context Window vs Context Length

### Critical Distinction

**Context Window:**
- The model's **maximum architectural capacity**
- Fixed by model design (positional embeddings, training)
- Example: Llama 3.2 3B has a **128K token** context window
- This is a **capability limit**, not memory usage

**Context Length (Sequence Length):**
- The **actual tokens used** in a specific request
- Example: User prompt (100 tokens) + generated response (50 tokens) = **150 tokens**
- This is what determines **memory consumption**
- Must be ≤ context window

```
Context Window = 128,000 tokens  ← What the model CAN handle
Context Length = 150 tokens      ← What you're ACTUALLY using
```

**Memory usage depends on actual context length, not window size.**

### Capacity Planning Reality

**Marketing vs Reality:**
- Marketing: "Supports 128K context!"
- Reality at 128K: 0.2606 MB/token × 128,000 = **33.4 GB per sample**
- That's more than entire GPU (24 GB)
- Max batch at 128K context = **0** samples (won't fit!)

**Practical capacity:**

| Context Length | Memory/Sample | Max Batch (with 2.5GB safety) |
|----------------|---------------|-------------------------------|
| 50 tokens | 13.03 MB | 1,200 |
| 2,048 tokens | 533.6 MB | 28 |
| 128,000 tokens | 33.4 GB | 0 (need multi-GPU) |

### Production Implications

**Pricing models:**
- Can't charge the same for 50-token vs 128K-token requests
- Memory/throughput cost scales linearly with actual usage
- Need per-token pricing, not per-request

**Multi-GPU requirement for long context:**

| Context Length | Memory/Sample | GPUs Needed per User |
|----------------|---------------|---------------------|
| 128K tokens | 33.4 GB | 2 GPUs (tensor parallel) |
| 64K tokens | 16.7 GB | 1 GPU (barely fits) |
| 32K tokens | 8.3 GB | 1 GPU (comfortable) |

---

## What Gets Stored in KV Cache

### What Is NOT Stored

**Attention weights (scores) are NOT stored!**

```
attention_weights = softmax(Q × K^T / √d)
```

These are computed fresh every time and then discarded.

### What IS Stored

The KV cache stores the **Key and Value projections** - intermediate representations from linear transformations, not final attention outputs.

### Step-by-Step Attention Process

**When processing token 1:**
```
1. Input embedding: x₁ [hidden_dim]

2. Project to Q, K, V using learned weights:
   Q₁ = x₁ × Wq  [num_heads × head_dim]
   K₁ = x₁ × Wk  [num_heads × head_dim]
   V₁ = x₁ × Wv  [num_heads × head_dim]

3. STORE K₁ and V₁ in cache ← THIS IS THE KV CACHE
   (Q₁ used immediately and discarded)

4. Compute attention output for token 1
```

**When generating token 50:**
```
1. Input: x₅₀

2. Project to Q, K, V:
   Q₅₀ = x₅₀ × Wq
   K₅₀ = x₅₀ × Wk
   V₅₀ = x₅₀ × Wv

3. LOAD K₁...K₄₉, V₁...V₄₉ from cache
   Append K₅₀, V₅₀ to cache

4. Compute attention (fresh, not cached):
   scores = softmax(Q₅₀ × [K₁...K₅₀]^T / √d)
   [How much token 50 attends to each previous token]

5. Compute output (fresh, not cached):
   output₅₀ = scores × [V₁...V₅₀]
   [Weighted sum of value vectors]

6. Discard Q₅₀ and scores (not stored)
   Keep K₅₀, V₅₀ in cache
```

### What K and V Represent

**K (Key) vectors:**
- Think of these as "addresses" or "tags" for each token
- When a new token computes attention, it asks "which previous tokens are relevant?"
- It does this by comparing its Query against all cached Keys

**V (Value) vectors:**
- Think of these as the "content" or "information" from each token
- Once attention weights determine relevance, the Values are combined
- The weighted sum of Values becomes the attention output

**Analogy:**
- K = labels on filing cabinet drawers
- V = contents inside those drawers
- Q = your search query
- Attention weights = which drawers are most relevant
- Output = information retrieved from relevant drawers

### The Efficiency Gain

**Without KV cache (recompute everything):**
```
Token 50 generation:
  Compute Q₁, K₁, V₁ (wasted)
  Compute Q₂, K₂, V₂ (wasted)
  ...
  Compute Q₅₀, K₅₀, V₅₀
  
Total: 50 × 3 projections = 150 matrix multiplications
```

**With KV cache:**
```
Token 50 generation:
  Load K₁...K₄₉, V₁...V₄₉ from memory (fast)
  Compute Q₅₀, K₅₀, V₅₀ (only new token)
  
Total: 3 matrix multiplications
```

**Speedup: 50× fewer computations!**

---

## Understanding K and V Projections

### What is a Projection?

A **projection** means applying a **learned linear transformation** (matrix multiplication) to convert one representation into another:

```
K = x × Wk
```

Where:
- `x` is the input (token embedding or hidden state)
- `Wk` is a learned weight matrix (frozen during inference)
- `K` is the resulting Key vector

### Concrete Example with "cat"

**Starting point:**
```
Token: "cat"
Token ID: 2574 (from vocabulary)

Embedding lookup:
embedding_table[2574] → [3,072-dimensional vector]

x = [0.234, -0.891, 0.445, ..., 0.123]  (3,072 numbers)
```

**Projection operation:**
```
Input: x [3,072 dimensions]

Learned weight matrices (from training):
  Wq: [3,072 × 3,072] - Query projection
  Wk: [3,072 × 3,072] - Key projection
  Wv: [3,072 × 3,072] - Value projection

Matrix multiplication:
  Q = x × Wq → [3,072 dim] reshaped to [24 heads × 128 dim/head]
  K = x × Wk → [3,072 dim] reshaped to [24 heads × 128 dim/head]
  V = x × Wv → [3,072 dim] reshaped to [24 heads × 128 dim/head]
```

**Result stored in KV cache:**
```
For token "cat" at layer 5, head 3:

K_cat = [0.892, -0.234, 0.567, ..., -0.123]  (128 numbers)
V_cat = [0.445, 0.778, -0.334, ..., 0.891]   (128 numbers)
```

### What Do These Numbers Mean?

They're **learned representations** - the model discovered during training that these transformations are useful:

- **K (Key):** Represents "what this token is about" in a way that can be compared
- **V (Value):** Represents "what information this token contains" to be retrieved

### Example: Attention in Action

**Scenario:** "The cat sat on the mat"

```
Cached K and V for previous tokens:

K_the = [0.12, 0.45, -0.23, ...]
V_the = [0.34, -0.12, 0.56, ...]

K_cat = [0.89, -0.23, 0.56, ...]
V_cat = [0.44, 0.78, -0.33, ...]

K_sat = [0.23, 0.67, -0.45, ...]
V_sat = [0.12, -0.34, 0.89, ...]
```

**Generating next token:**
```
1. New token's Query:
   Q_new = [0.45, 0.78, -0.12, ...]

2. Compare with cached Keys (dot product):
   similarity_to_the = Q_new · K_the = 0.34
   similarity_to_cat = Q_new · K_cat = 0.89  ← high!
   similarity_to_sat = Q_new · K_sat = 0.45

3. Attention weights (softmax):
   weight_the = 0.15
   weight_cat = 0.62  ← "cat" is most relevant
   weight_sat = 0.23

4. Weighted sum of Values:
   output = 0.15×V_the + 0.62×V_cat + 0.23×V_sat + ...
```

### Why Three Different Projections?

The three projections let the model learn **different roles**:

```
Same input: x_cat = [token embedding]

Wq transforms it for "asking questions":
  Q_cat asks: "What tokens should I pay attention to?"

Wk transforms it for "being found":
  K_cat answers: "I'm relevant when you're asking about animals"

Wv transforms it for "providing information":
  V_cat contains: "Information about this specific cat"
```

### Storage Details

**For one token, one layer in cache:**
```
K tensor: [24 heads × 128 dim] = 3,072 FP16 values × 2 bytes = 6,144 bytes
V tensor: [24 heads × 128 dim] = 3,072 FP16 values × 2 bytes = 6,144 bytes

Total: 12,288 bytes per token per layer
Across 28 layers: 344 KB per token
```

### Learned vs Computed

**Weight matrices (Wq, Wk, Wv):**
- Learned during training
- Frozen during inference
- Part of model weights (6.0 GB base)
- Same for all tokens

**Projections (Q, K, V):**
- Computed during inference
- Different for each token
- K and V stored in cache (reusable)
- Q computed fresh (not reusable)

---

## Interview Articulations

### Two Bottlenecks

"Week 1 revealed two distinct bottlenecks. First, memory bandwidth limits single-request performance - FP16 only gave 1.56x speedup because autoregressive decode is memory-bound, achieving 54% of peak bandwidth while tensor cores sit idle. Second, software inefficiency limits batch scaling - the transformers library's throughput plateaued at 5,000 tok/s with per-sample performance collapsing 95% from 84 to 4.2 tok/s. The key distinction: if memory bandwidth were the issue, larger batches would help by amortizing weight loads. The collapse proves this is a software parallelization problem, validating the need for production frameworks like vLLM."

### Memory Bandwidth and Batching

"Batching should improve bandwidth efficiency by amortizing weight loads across multiple samples. With batch=1, you load 6GB of weights to compute one output. With batch=1200, you load 6GB once to compute 1200 outputs. The fact that per-sample throughput collapsed by 95% proves this is a software parallelization issue, not a hardware bandwidth constraint. The transformers library cannot effectively coordinate 1200 concurrent token generations even though the GPU has the capacity."

### KV Cache Mechanism

"The KV cache transforms autoregressive generation from O(n²) to O(n) complexity by storing reusable Key and Value projections. When generating token 50, instead of recomputing keys and values for all 49 previous tokens, we load them from cache and only compute projections for the new token - a 50× reduction in matrix multiplications. This is why KV cache is critical for production inference efficiency."

### Architecture and Memory

"KV cache memory is determined by the transformer's architectural parameters: layers, attention heads, and head dimension. For Llama 3.2 3B with 28 layers and 24 heads of 128 dimensions each, every token requires 344KB of FP16 storage. This scales linearly with sequence length and batch size, meaning a 2K-token context uses 688 MB per request. Understanding this relationship is critical for capacity planning: doubling model depth doubles KV cache requirements, and the choice of multi-head architecture directly impacts serving costs."

### Batch Size vs Sequence Length

"Batch size and sequence length are independent dimensions that both scale KV cache memory linearly. A production inference system must handle the 2D optimization problem: maximize batch size for throughput while supporting variable sequence lengths for user experience. The product of these dimensions determines memory consumption: doubling either dimension doubles memory usage. This is why frameworks like vLLM that efficiently manage variable-length batches are critical for production deployments."

### Context Window vs Context Length

"The context window is the model's architectural maximum - like RAM capacity. Context length is actual usage - like active memory consumption. A model with a 128K context window still consumes memory proportional to actual usage. For production capacity planning, you must analyze real user behavior - a model supporting 128K tokens might only serve one concurrent user at that length, but could serve 1,200 users at 50 tokens each. This creates a direct tradeoff between context length and throughput that drives both infrastructure sizing and pricing models."

### What's Stored in KV Cache

"The KV cache stores the Key and Value projections for each token at each layer - not the attention weights themselves. These projections are intermediate representations that enable efficient reuse: when generating a new token, we compute its Query vector and combine it with cached Keys to calculate fresh attention scores. Those scores then weight the cached Values to produce the attention output. The attention weights are ephemeral - computed and discarded each step - but the K and V tensors are reusable across all future tokens, which is why we cache them. This transforms autoregressive generation from O(n²) recomputation to O(n) incremental computation."

### K and V Projections

"The K and V projections are learned linear transformations applied to token embeddings. Each token's embedding - a 3,072-dimensional vector - is multiplied by three separate weight matrices (Wq, Wk, Wv) that were learned during training. The resulting K and V tensors are 3,072-dimensional vectors reshaped into 24 heads of 128 dimensions each. These aren't just copies of the embedding - they're transformed representations where Keys encode 'what to match against' and Values encode 'what information to retrieve.' For Llama 3.2 3B, storing these two 3,072-dimensional FP16 tensors per token per layer requires 12,288 bytes, which across 28 layers totals 344KB per token."

### Production Framework Implications

"Vanilla PyTorch's transformers library is unsuitable for production inference at scale. While memory analysis shows theoretical capacity for 1,200 concurrent users, batch scaling plateaus at 5,000 tok/s due to poor parallelization. Production frameworks like vLLM solve this through continuous batching, PagedAttention for efficient KV cache management, and fused CUDA kernels. The expected improvement is 50-60 tok/s per sample even at batch 512, versus 9.2 tok/s with transformers - a 5-6× throughput gain at the same hardware utilization."

### Real-World Capacity Planning

"Memory-based capacity planning gives 1,200 users per GPU, but throughput-based planning reveals only 100-150 users due to framework limitations. The optimal batch size is 64-128, not 1,200, because beyond that point you get minimal throughput gains while destroying per-user experience. Additionally, context length creates a multiplicative impact: going from 50 to 2,048 tokens reduces capacity from 1,200 to 28 users per GPU - a 42× reduction. Production planning must account for actual usage patterns, framework efficiency, and the context length-throughput tradeoff."

---

## Key Production Takeaways

1. **Framework choice is more critical than hardware** for inference optimization
2. **KV cache memory scales linearly** with both sequence length and batch size
3. **Context window ≠ capacity** - actual usage determines memory consumption
4. **Batch scaling has diminishing returns** - optimal range is 64-128, not maximum possible
5. **Memory bandwidth and software efficiency are separate bottlenecks** requiring different optimizations
6. **Production capacity is throughput-limited**, not memory-limited, with current frameworks

These insights validate the 24-week training plan structure: Phase 1 establishes baseline understanding, Phase 2 introduces production frameworks (vLLM, Triton) that solve the exact limitations discovered in Week 1.

---

*Document created: Week 1 completion*  
*Hardware: 2x RTX 3090, Ubuntu 24.04, CUDA 12.6*  
*Model tested: Llama 3.2 3B Instruct*
