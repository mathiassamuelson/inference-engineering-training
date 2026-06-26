# Large Language Models, Decomposed

A compendium that takes the LLM apart one mechanism at a time — built from working through each
piece on a single concrete model, with the numbers that make the mechanics stick.

## Scope and conventions

- **Single model throughout.** Every entry uses the **Gemma 4 12B** model from this repo's
  inference-stack work (hidden_size 3840, ~15360 FFN intermediate, 48 transformer blocks, w4a16
  weights). Numbers stay consistent across entries so the reader builds intuition without
  context-switching between architectures. Where a value is illustrative or architecture-dependent
  rather than a hard spec, entries say so.
- **Strictly mechanical.** Entries focus on how the machinery works. Tangential or editorial
  material (e.g. the state of interpretability as a research field) is kept out, or clearly marked
  as an aside, so the core reference stays clean.
- **"Block" vs "layer."** "Block" = the repeating transformer unit (attention + FFN + norms +
  residuals); "layer" is reserved for primitive operations (a linear layer, a norm layer). The
  field often overloads "layer" for both scales; this is a deliberate local convention for clarity.
- **Concrete over abstract.** Mechanisms are made tangible with small worked examples (e.g. by-hand
  matmuls) before scaling up to the real dimensions.
- **Honest about uncertainty.** Implementation- or convention-dependent claims are flagged rather
  than asserted as universal.

## Naming

Entries follow `entry-NN-<topic-slug>.md` — numbered for order, slugged for scanability.

## Entries

| # | Entry | Covers |
|---|---|---|
| 01 | [The Feed-Forward Network, the Shapes of Inference, and Tensor Parallelism](entry-01-ffn-inference-shapes-tensor-parallelism.md) | Matrix-multiply mechanics and the row-first convention; the FFN (up/down projection, GeLU, why expand-then-contract); weights vs. activations and what's learned; the `(batch × sequence × features)` tensor; prefill (compute-bound) vs. decode (bandwidth-bound); the KV cache; tensor parallelism (column/row split, the all-reduce and its byte budget); differentiability and why everything must be differentiable. |

## Planned / in progress

- **Entry 02 — Attention.** The cross-token mixing step: Q/K/V projections, attention scores →
  softmax → weighted sum, the causal mask and RoPE, multi-head vs. grouped-query attention, and how
  the KV cache and the tensor-parallel head-split connect back to Entry 01. (Pickup notes for this
  entry live at the end of Entry 01.)
