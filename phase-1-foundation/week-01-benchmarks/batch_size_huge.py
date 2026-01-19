import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Test limit
for batch_size in [1200, 1400, 1600, 1800]:
    try:
        print(f"Testing batch={batch_size}...")
        torch.cuda.reset_peak_memory_stats(0)
        prompts = ["GPU memory:" for _ in range(batch_size)]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda:0")
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.pad_token_id)
        peak = torch.cuda.max_memory_allocated(0) / 1e9
        print(f"  ✓ Success! Peak: {peak:.2f} GB")
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"  ✗ OOM at batch={batch_size}")
            break
        raise