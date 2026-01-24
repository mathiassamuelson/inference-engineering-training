#!/usr/bin/env python3
"""
Experiment 2, Step 1: Export Llama 3.2 3B to ONNX
This will reveal the challenges of LLM conversion
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

print("=" * 70)
print("EXPERIMENT 2, STEP 1: LLAMA 3.2 3B ONNX EXPORT")
print("=" * 70)

# Configuration
model_name = "meta-llama/Llama-3.2-3B-Instruct"
output_dir = "results/llama_onnx"
os.makedirs(output_dir, exist_ok=True)

# Load model
print("\n[1/4] Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="cuda:0"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

print(
    f"  ✓ Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B parameters"
)
print(f"  ✓ Memory allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

# Prepare dummy input for export
print("\n[2/4] Preparing export inputs...")
sequence_length = 128  # Fixed for initial export
batch_size = 1

dummy_input = torch.randint(
    0,
    tokenizer.vocab_size,
    (batch_size, sequence_length),
    dtype=torch.long,
    device="cuda:0",
)
attention_mask = torch.ones(
    (batch_size, sequence_length), dtype=torch.long, device="cuda:0"
)

print(f"  ✓ Input shape: {dummy_input.shape}")
print(f"  ✓ Sequence length: {sequence_length}")

# Attempt ONNX export
print("\n[3/4] Attempting ONNX export...")
print("  (This may take 2-5 minutes and use significant memory)")

onnx_path = os.path.join(output_dir, "llama_3b.onnx")

try:
    # Standard ONNX export - this often fails for LLMs
    torch.onnx.export(
        model,
        (dummy_input, attention_mask),
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
    )

    # Check file size
    file_size_gb = os.path.getsize(onnx_path) / 1e9
    print(f"\n  ✓ ONNX export SUCCESS!")
    print(f"  ✓ File: {onnx_path}")
    print(f"  ✓ Size: {file_size_gb:.2f} GB")

except Exception as e:
    print(f"\n  ✗ ONNX export FAILED")
    print(f"  Error type: {type(e).__name__}")
    print(f"  Error message: {str(e)[:500]}")
    print("\n" + "=" * 70)
    print("EXPECTED FAILURE - LLM EXPORT CHALLENGES")
    print("=" * 70)
    print("""
Common issues with direct Llama ONNX export:

1. **KV Cache Not Supported**
   - ONNX export captures a single forward pass
   - Autoregressive generation requires stateful KV cache
   - Each new token needs access to cached keys/values

2. **Dynamic Shapes Complexity**
   - Sequence length changes during generation
   - KV cache grows with each token
   - ONNX dynamic axes don't handle this well

3. **Unsupported Operations**
   - RoPE (rotary position embeddings) may not export cleanly
   - Some attention implementations use torch-specific ops

4. **Memory Explosion**
   - ONNX graph can be much larger than PyTorch model
   - 3B model → potentially 10+ GB ONNX file

This is WHY TensorRT-LLM exists - it handles these complexities.
""")

# Memory cleanup
print("\n[4/4] Cleanup...")
del model
torch.cuda.empty_cache()
print(f"  ✓ GPU memory freed")

print("\n" + "=" * 70)
print("STEP 1 COMPLETE")
print("=" * 70)
