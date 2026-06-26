# Large Language Models, Decomposed

## Entry 1 — The Feed-Forward Network, the Shapes of Inference, and Tensor Parallelism

*A reference built from working through the mechanics on a concrete model. Where a number
illuminates, it's included; where a claim is convention- or implementation-dependent, that's
flagged. The running example throughout is the Gemma-style 12B model from the inference-stack
work.*

---

### 0. The running example

All concrete numbers below use the 12B model under study:

- **hidden_size = 3840** — the model's "working width." Every token, between blocks, is a
  3840-number vector.
- **intermediate width ≈ 15360** — the FFN's wide middle (roughly 4× the working width; exact
  ratio is an architecture choice).
- **48 transformer blocks** — the repeating unit, stacked 48 deep.
- Quantized to **4-bit weights (w4a16)**: ~0.5 byte per parameter for the weights; activations
  flow at higher precision (BF16, 2 bytes).

Terminology note carried throughout: **"block"** = the repeating transformer unit (attention +
FFN + norms + residuals); **"layer"** is reserved for primitive operations (a linear layer, a norm
layer). "The model has 48 blocks" and "the FFN has 2 linear layers" then read unambiguously. The
field often says "layer" for both scales (the config key is literally `num_hidden_layers`), so this
is a deliberate local convention for clarity, not universal usage.

---

### 1. Matrix multiplication — the load-bearing operation

Nearly everything a transformer does at runtime is matrix multiplication. The mechanics are worth
holding precisely because every shape, cost, and parallelization trick downstream is a consequence
of them.

**The rule (convention-independent).** For `(a × b) · (b × c) → (a × c)`: the two **adjacent
inner** dimensions (the `b`'s, touching across the `·`) must match and cancel; the two **outer**
dimensions (`a`, `c`) survive as the result shape. The rule is about *adjacency*, not about which
operand is the "matrix." Stated most generally: **the dimensions that touch across the multiply
cancel; the dimensions on the outside remain.**

**Row-first convention.** Deep learning writes `x · W` (activation on the left, weight matrix on
the right), not the textbook `W · x`. With `x` as a row vector `(1 × 3840)` and `W` as
`(3840 × 15360)`: inner 3840's cancel, outer 1 and 15360 survive → `(1 × 15360)`. The two
conventions produce transposes of each other (same numbers, stood up as a column vs. laid out as a
row).

*Why deep learning chose row-first:* batching. Stack many tokens as **rows** of a matrix
`X = (batch × features)`, and `X · W` processes the whole batch in one multiply, output
`(batch × out)` — one row per token, batch on the leading axis, which is how every tensor in the
framework is shaped (`[batch, features]`). The textbook column convention would put batch on the
trailing axis, which is awkward. PyTorch's `nn.Linear` stores its weight and computes `x @ W.T`
under the hood; the convention is baked in.

**A "vector" is just a thin matrix.** A row vector is a `(1 × n)` matrix — no special object, no
special rules. This collapses a lot of confusion: the jump from one token `(1 × 3840)` to a batch
`(32 × 3840)` is just changing the leading dimension from 1 to 32; *nothing else about the
operation changes*. The general object is a **tensor**, and scalar / vector / matrix are tensors of
rank 0 / 1 / 2 (increasing numbers of axes). The actual shape flowing through a transformer is
rank-3: `(batch × sequence × features)` — see §5.

**Where each output comes from.** In `x · W`, output number `j` is `x` dotted against **column `j`**
of `W`. This single fact is the seed of the tensor-parallelism split in §6: each output "owns" one
column of weights, so splitting `W` by columns gives independent outputs.

*Worked example (row-first):* `[2 3 4] · W` where W's columns are `[1,2,0], [0,1,4], [5,0,1]` →
`[8, 19, 14]` (each output is the input dotted with one column). A batch stacks independent rows;
each row is processed identically and independently — `(3×3)·(3×3)` gives three independent
row-results.

---

### 2. The Feed-Forward Network (FFN / MLP)

Each block has two sub-parts: attention (mixes information *between* tokens — Entry 2) and the FFN
(per-token feature processing). The FFN is two matmuls with a nonlinearity between them:

```
  x (3840)  --W_up-->  h (15360)  --GeLU-->  h' (15360)  --W_down-->  out (3840)
            up-proj                          (elementwise)  down-proj
```

**Why up-then-down (expand, then contract).** The expansion isn't the point on its own; it gives
the nonlinearity a wide space to operate in. Each of the ~15360 intermediate units is a learned
"feature detector"; the up-projection computes them, GeLU gates each one, the down-projection
recombines the survivors. More intermediate width = more independent features the block can detect
and gate. It **must** return to 3840 because blocks stack (output of block *i* is input of *i+1*)
and a residual connection adds the block's input back to its output — both require identical widths.
So: *expand to do the work, contract to honor the width contract.* The expansion ratio (here ~4×)
is a tuned capacity knob.

**Parameter weight.** One block's FFN is `3840×15360 + 15360×3840 ≈ 118M` parameters; ×48 blocks
≈ **5.7B** — roughly half the model. The FFN is where most parameters (and most "knowledge") live.

---

### 3. Nonlinearity and GeLU

**Why a nonlinearity is mandatory.** A matmul is linear, and stacking linear operations collapses
to a single linear operation: `(x·W1)·W2 = x·(W1·W2)`. A network of only matmuls, however deep,
is equivalent to one matmul — it can only represent straight-line relationships. A **non-linear**
function between the matmuls breaks that collapse and is what gives depth its power.

**GeLU** (Gaussian Error Linear Unit) is the per-element nonlinear gate used here (Gemma uses a
tanh-approximation variant). Behavior: large positive input passes through ≈ unchanged; large
negative is suppressed toward 0; near zero, a smooth ramp. It's essentially a **smoothed ReLU**
(`max(0,x)`, which has a hard corner). Relatives: SwiGLU, used by many models — same role.

**Key property used in §6:** GeLU is **elementwise** — each output depends only on the single input
in the same position. This is why, under tensor parallelism, each card can apply GeLU to its own
shard of the intermediate with *no communication*.

---

### 4. Weights vs. activations — what's learned, what's computed

A clean and load-bearing dividing line:

- **Weights** (`W_up`, `W_down`, the attention projection matrices, the embedding, the LM head):
  **learned** during training, **frozen** after, **identical for every request**, read from memory
  every forward pass. These are the gigabytes that get read (and split across cards in §6).
- **Activations** (the intermediate `(1, 15360)` vector, the hidden states): **computed fresh** at
  runtime from the input, **different every request**, never stored, recomputed every pass. These
  are the small things that flow through (and get all-reduced in §6).

*Learned = frozen weight. Computed = ephemeral activation.* The "weight read per token" that
dominates decode cost (§5) and that TP=2 halves (§6) is precisely the cost of fetching these
**learned** matrices out of memory.

**Not everything is learned.** Fixed architectural scaffolding — the positional encoding scheme
(RoPE), the causal attention mask, the activation function itself — is defined by the architecture,
not fit to data. So the precise statement is: *every weight matrix is learned; some structural
scaffolding is fixed.*

---

### 5. The shapes of inference — batch, sequence, prefill, decode

The real tensor flowing through a transformer is rank-3: **`(batch × sequence × features)`**.

- **features (3840):** what each token *is* — the axis the matmuls operate on.
- **sequence (S):** how many tokens in a row within one request.
- **batch (B):** how many independent requests.

The matmul operates on the last axis (`features`); **all leading axes ride along**. `W_up` just
hits the 3840 and processes however many 3840-vectors are stacked in front of it.

**Two regimes, distinguished entirely by the sequence axis:**

| | shape (per request) | work per weight-read | bottleneck |
|---|---|---|---|
| **Prefill** | `(1, S=prompt_len, 3840)` | S rows (e.g. 8192) | **compute-bound** — matmul units saturated |
| **Decode** | `(1, S=1, 3840)` | 1 row | **bandwidth-bound** — paying to read weights for one token |

Same `W_up`, same weight-read cost in both — but prefill amortizes it across thousands of tokens
while decode amortizes across one. **This is why prefill throughput (~2480 tok/s measured) vastly
exceeds decode (~70 tok/s):** prefill keeps the arithmetic busy; decode leaves it idle, waiting on
memory. The `S` axis being large vs. 1 *is* the compute-bound vs. bandwidth-bound distinction,
expressed as a shape.

**The KV cache is what makes decode S=1.** Without it, generating the 200th token would mean
re-processing all prior tokens every step. The cache stores each past token's K and V (computed
once, during the pass that first saw it), so a new decode step computes only its *own* K and V,
appends them, and attends against the stored rest — collapsing the work to one new token. Cache
growth ≈ `tokens × num_blocks × num_kv_heads × head_dim × 2 (K and V)` — note the **×num_blocks**:
it's per-block, so it scales with sequence length *and* depth, which is why long contexts eat GPU
memory. (vLLM's memory model: KV cache is the residual capacity left after weights, CUDA graphs,
and overhead.)

**Phase boundary nuance:** the prefill pass also produces the *first* output token (it computes the
logits for the token after the prompt). Decode continues from the second output token. So it's
"prefill fills the cache *and* emits token 1, then decode continues," not a clean stop/start.

**Batching and concurrency.** Stacking the **batch axis B** (multiple requests) amortizes the fixed
weight-read across more tokens. This helps **decode** enormously (it's memory-starved, desperate
for amortization) and **prefill** little (already compute-saturated). Measured TP=1 aggregate
decode climbed 70 → 112 tok/s across c=1→8 as the weight-read amortized — while *per-request*
decode fell 70 → 16 (the requests share one GPU's fixed resources).

*How batching actually works in production (vLLM):* not the clean `(B, S, features)` rectangle.
- **Continuous (in-flight) batching:** the batch is assembled *per step* from whichever sequences
  are active; requests at different positions in different conversations are fused into one
  `(active_tokens, 3840)` matmul. The batch composition changes every step as requests finish/join.
- **Ragged batching (no padding):** real prompts differ in length; rather than pad to a rectangle
  (wasteful), vLLM concatenates real tokens into one flat `(Σtokens, 3840)` tensor with an index
  tracking sequence boundaries (variable-length FlashAttention keeps sequences from bleeding).
- **Chunked prefill:** because prefill is compute-bound and decode is memory-bound, vLLM slots
  prefill chunks *alongside* other requests' decode tokens in one batch — the two use different
  bottleneck resources, so together they use the GPU more fully than either alone.

*A controlled benchmark (like the TP sweep) deliberately isolates prefill and decode into clean
phases; production deliberately mixes them. Both are correct for their purpose.*

---

### 6. Tensor parallelism — splitting a model across GPUs

Tensor parallelism (TP=2) splits each weight **matrix** across two cards. The activation is
**replicated** (full copy on both cards) at block boundaries and **sharded** only in the middle —
the art is arranging splits so the activation is replicated exactly where the next operation needs
it, and sharded everywhere it's free.

**The column-then-row scheme (NVIDIA's Megatron-LM).** Two ways to split a weight matrix produce
two very different situations:

- **Column-split** (applied to `W_up`): each card gets half the *columns*, both cards get the full
  input. Each card computes **complete** output columns (a finished half of the intermediate). To
  combine: **concatenate** — free, no communication. (Because output `j` owns column `j`; split the
  columns, split the outputs cleanly.) GeLU then runs elementwise on each card's half — still free.
- **Row-split** (applied to `W_down`): each card gets half the *rows*, matching the intermediate's
  split. Splitting along the **shared/inner** dimension means each card sums over only *its* half →
  each produces a **full-width partial sum**. To combine: **add** the two partials elementwise. That
  addition is the **all-reduce** — one communication, at the very end.

Chaining column-then-row means the wide intermediate never gets assembled or moved; all
communication defers to a single point. **One all-reduce per FFN, one per attention block → two per
transformer block.** Doing it row-then-column would force a sync in the *middle* and at the end —
two syncs. The ordering is chosen to minimize communication.

*(Caveat: "two all-reduces per block" is the Megatron-standard forward-pass count; specific
implementations and attention variants can change it. Mechanism is robust; exact count is
implementation-dependent.)*

**What an all-reduce is.** A collective operation: every participant contributes a value, the values
are combined (summed), and **every** participant gets the total back. The "all" matters — both cards
need the full result to start the next step. (Contrast plain *reduce*, where only one participant
gets the result — that's the MapReduce-style funnel. All-reduce broadcasts to everyone.) It is *not*
MapReduce: MapReduce is a coarse, fault-tolerant, disk-spilling batch framework; all-reduce is a
microsecond-scale lockstep primitive run hundreds of times per second. They share only the
functional-programming root meaning of "reduce" (combine many → fewer via an associative operator).

**Ring all-reduce** is the bandwidth-optimal implementation: arrange participants in a ring, pass
data around in chunks (a reduce-scatter pass then an all-gather pass), so no single card is a
bottleneck and the bytes each card moves don't grow with participant count. (Baidu popularized the
DL version; NVIDIA's NCCL library implements it; for N=2 it degenerates to a swap-and-add.)

**The byte budget — why it's cheap on NVLink.** The all-reduce moves the *activation*, not the
weights:
- One all-reduce ≈ one hidden-state vector = 3840 × 2 bytes ≈ **7.5 KB** (~15 KB counting the
  round-trip).
- Per token: 2 per block × 48 blocks = ~96 all-reduces ≈ **1.4 MB of traffic per token**.
- The weight-read it *saves*: ~gigabytes per token (TP=2 halves each card's read). **Ratio ≈
  2,000 : 1** — the thing TP adds is ~1000× smaller than the thing it removes.

This is why the empirical finding (TP=2 on the NVLink pair, +47% c=1 decode) "stops looking
surprising and starts looking inevitable": you trade gigabytes of memory reads for a megabyte of
link chatter. **The result is interconnect-dependent** — on a slow link (e.g. PCIe x1, ~1 GB/s) the
per-token all-reduce would dominate and the same split would lose. The win rests on the all-reduce
being cheap, which is true only on NVLink.

*Connection to the residual decode penalty:* the byte-ratio alone would predict a near-2× win; the
measured +47% is smaller because the all-reduce is so tiny it's **latency-bound, not
bandwidth-bound** — the fixed cost of launching 96 sync operations per token (not their bytes)
plus the KV-read and per-step overhead that don't halve. That overhead is the gap between +47% and
the +100% ceiling.

---

### 7. Features and interpretability (aside)

The "features" the FFN detects are **not** the hand-engineered, named features of classical ML
(rooms, square-feet). They are **learned, anonymous, emergent** — the up-projection's weights settle
during training into units that respond to *some* useful pattern, named by no one. Most don't map to
a clean human concept; some legibly track things (a "this is French" unit); and crucially features
are often **not** one-unit-one-concept but smeared across many units (**superposition** — more
features than units, stored as overlapping directions). Recovering what these directions mean,
after the fact, is the field of **mechanistic interpretability** — real and causal (you can clamp a
feature and change behavior, e.g. "Golden Gate Claude"), but early, partial, and with its biggest
safety claims still promissory. Worth triangulating proponents against critics.

---

### 8. How training reaches back to the weights — differentiability

Training adjusts weights by **backpropagation** = the **chain rule** across the whole network. To
update `W_up`, you need the gradient of the loss w.r.t. `W_up`, computed by multiplying the local
derivatives of every operation between `W_up` and the loss. **If any operation on that path has no
derivative, the gradient can't flow** — so every operation must be differentiable.

- Matmuls: trivially differentiable (linear).
- GeLU: smooth everywhere — differentiable, a selling point.
- ReLU: has a **corner at 0** where the derivative is undefined — yet trains fine, because (a) the
  bad point is **measure-zero** (activations essentially never land exactly on 0) and (b) frameworks
  use a **subgradient convention** (PyTorch hardcodes `relu'(0)=0`). So the precise requirement is
  **differentiable *almost everywhere*, with a subgradient convention covering the gaps.**
- GeLU's smoothness over ReLU helps optimization: ReLU's hard corner causes "dying ReLU" (a unit
  pushed all-negative has gradient 0 forever, stuck) and a discontinuous gradient; GeLU's gradient
  is continuous and small-but-nonzero for negatives, so units don't get permanently stuck.

**The general principle (one of the deepest organizing constraints in deep learning):** *everything
is built to be differentiable almost everywhere, because gradient descent is the only known way to
train at scale.* This is why attention uses **softmax** (a smooth, differentiable approximation of
"pick the max") rather than a hard max; why discrete operations get differentiable relaxations
during training and are hardened only at inference. The whole design space is implicitly filtered
to "things you can take the gradient of."

---

## Next entry — Attention (pickup for the fresh chat)

The natural next decomposition. It will extend, not replace, everything above — attention is the
*other* sub-part of each block (the FFN's sibling), and it reuses machinery already understood here.

**Already established (don't re-derive — build on these):**
- Matmul mechanics, row-first convention, the inner/outer rule, tensor as ranked object.
- `(batch × sequence × features)`, prefill (large S, compute-bound) vs. decode (S=1,
  bandwidth-bound).
- The **KV cache**: what it stores and why it makes decode S=1 — attention is *where* K and V come
  from and what they're for, so this is the direct entry point.
- Weights-vs-activations; the column-split/row-split TP scheme and the all-reduce.
- Differentiability; softmax as the differentiable "pick-the-max."

**Hooks to open attention on (where it connects to what's known):**
- Attention is the **cross-token mixing** step (the FFN does per-token processing; attention is
  where tokens *look at each other*). This is the one thing the FFN explicitly does *not* do.
- The **Q, K, V projections** are the learned weight matrices that turn each token's 3840-vector
  into a query, a key, and a value. In the TP scheme these are **column-split** (the natural unit is
  the attention **head** — heads are independently parallel), and the **output projection** that
  recombines heads is **row-split** → the same one-all-reduce-per-sub-block pattern as the FFN.
- The **KV cache** is exactly the stored K and V from this mechanism — so the cache discussion from
  Entry 1 §5 plugs straight in.
- **softmax** (already flagged as the differentiable max) is the core of how attention weights are
  computed — its differentiability and its numerical behavior are worth a close look.
- Variants worth reaching: **multi-head** vs. **grouped-query attention (GQA)** — GQA shrinks
  `num_kv_heads` below `num_attention_heads`, which is *why* the KV-cache size formula in §5 has a
  separate `num_kv_heads` term, and is a major lever on cache memory.
- The **causal mask** (already noted as fixed, non-learned scaffolding) and **RoPE** positional
  encoding — both live in attention.

**Suggested framing for the attention entry:** start from "what problem does attention solve that
the FFN can't" (cross-token information flow), build Q/K/V from the matmul mechanics already known,
make the attention-scores → softmax → weighted-sum pipeline concrete with small numbers (as the FFN
was), then connect to the KV cache and the TP head-split. Keep the "block" vs "layer" convention.
