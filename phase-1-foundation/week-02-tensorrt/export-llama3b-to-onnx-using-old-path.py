#!/usr/bin/env python3
"""
Experiment 2, Step 2: Legacy ONNX Export Attempt
Trying older export method that's more permissive
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

print("=" * 70)
print("EXPERIMENT 2, STEP 2: LEGACY ONNX EXPORT")
print("=" * 70)

model_name = "meta-llama/Llama-3.2-3B-Instruct"
output_dir = "results/llama_onnx"
os.makedirs(output_dir, exist_ok=True)

# Load model with specific settings for export
print("\n[1/5] Loading model with export-friendly settings...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda:0",
    use_cache=False,  # Disable KV cache for export
    attn_implementation="eager",  # Use eager attention, not flash/sdpa
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

print(
    f"  ✓ Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B parameters"
)
print(f"  ✓ KV cache: DISABLED (required for ONNX)")
print(f"  ✓ Attention: eager mode (not flash attention)")

# Prepare fixed-size input
print("\n[2/5] Preparing export inputs...")
sequence_length = 64  # Smaller for faster export
batch_size = 1

dummy_input_ids = torch.randint(
    0,
    tokenizer.vocab_size,
    (batch_size, sequence_length),
    dtype=torch.long,
    device="cuda:0",
)
dummy_attention_mask = torch.ones(
    (batch_size, sequence_length), dtype=torch.long, device="cuda:0"
)

print(f"  ✓ Input IDs shape: {dummy_input_ids.shape}")
print(f"  ✓ Attention mask shape: {dummy_attention_mask.shape}")

# Create wrapper to simplify export
print("\n[3/5] Creating export wrapper...")


class LlamaForExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False,
        )
        # Return only logits (first element)
        return outputs[0]


export_model = LlamaForExport(model)
export_model.eval()

# Test forward pass first
print("\n[4/5] Testing forward pass...")
with torch.no_grad():
    test_output = export_model(dummy_input_ids, dummy_attention_mask)
print(f"  ✓ Forward pass successful")
print(f"  ✓ Output shape: {test_output.shape}")
print(f"  ✓ Output dtype: {test_output.dtype}")

# Attempt legacy ONNX export
print("\n[5/5] Attempting LEGACY ONNX export...")
print("  (This bypasses torch.export, using older tracing method)")
print("  (May take 3-10 minutes...)")

onnx_path = os.path.join(output_dir, "llama_3b_legacy.onnx")

try:
    with torch.no_grad():
        torch.onnx.export(
            export_model,
            (dummy_input_ids, dummy_attention_mask),
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 1: "seq_len"},
                "logits": {0: "batch", 1: "seq_len"},
            },
            verbose=False,
        )

    file_size_gb = os.path.getsize(onnx_path) / 1e9
    print(f"\n  ✓ ONNX export SUCCESS!")
    print(f"  ✓ File: {onnx_path}")
    print(f"  ✓ Size: {file_size_gb:.2f} GB")

    # Validate ONNX model
    print("\n  Validating ONNX model...")
    import onnx

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"  ✓ ONNX validation passed")

except Exception as e:
    print(f"\n  ✗ Legacy export also FAILED")
    print(f"  Error type: {type(e).__name__}")
    error_msg = str(e)
    # Print first 1000 chars of error
    print(f"  Error: {error_msg[:1000]}")

    if len(error_msg) > 1000:
        print(f"  ... (truncated, {len(error_msg)} total chars)")

# Cleanup
print("\n" + "-" * 70)
print("Cleanup...")
del model, export_model
torch.cuda.empty_cache()
print("✓ GPU memory freed")

print("\n" + "=" * 70)
print("STEP 2 COMPLETE")
print("=" * 70)
print("""
NEXT STEPS based on result:

If SUCCESS:
  → Proceed to TensorRT conversion
  → But note: This only works for single forward pass (no generation)
  → Text generation requires KV cache which we disabled

If FAILED:
  → This confirms LLMs need specialized tooling
  → We'll use TensorRT-LLM instead (production approach)
  → This is the same conclusion NVIDIA reached
""")
