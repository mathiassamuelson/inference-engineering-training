"""
Week 3 - Experiment 3: Pipeline Parallelism Performance Analysis
Measures the cost of pipeline parallelism over PCIe x1 interconnects.
Uses Llama 3.1 8B which fits on 1 GPU (baseline) but can be forced across multiple.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import gc

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 50
WARMUP_ITERATIONS = 3
BENCHMARK_ITERATIONS = 10


def get_layer_count(model):
    """Get number of transformer layers in the model."""
    if hasattr(model.model, 'layers'):
        return len(model.model.layers)
    elif hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        return len(model.model.decoder.layers)
    return None


def build_device_map(model_name, num_gpus, gpu_ids):
    """
    Build an explicit device map that distributes layers across specified GPUs.
    This forces pipeline parallelism by placing sequential layer groups on different GPUs.
    """
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    num_layers = config.num_hidden_layers  # 32 for Llama 3.1 8B

    layers_per_gpu = num_layers // num_gpus
    remainder = num_layers % num_gpus

    device_map = {}

    # Embedding and input layers go on first GPU
    device_map["model.embed_tokens"] = gpu_ids[0]
    device_map["model.norm"] = gpu_ids[-1]
    device_map["model.rotary_emb"] = gpu_ids[0]
    device_map["lm_head"] = gpu_ids[-1]

    # Distribute transformer layers across GPUs
    layer_idx = 0
    for gpu_rank, gpu_id in enumerate(gpu_ids):
        # Give one extra layer to early GPUs if remainder exists
        n_layers = layers_per_gpu + (1 if gpu_rank < remainder else 0)
        for _ in range(n_layers):
            device_map[f"model.layers.{layer_idx}"] = gpu_id
            layer_idx += 1

    return device_map, num_layers


def print_device_map_summary(device_map, num_layers, gpu_ids):
    """Print which layers are on which GPU."""
    gpu_layers = {gpu_id: [] for gpu_id in gpu_ids}
    for key, gpu_id in device_map.items():
        if key.startswith("model.layers."):
            layer_num = int(key.split(".")[-1])
            gpu_layers[gpu_id].append(layer_num)

    print(f"  Pipeline configuration ({num_layers} layers):")
    for gpu_id in gpu_ids:
        layers = sorted(gpu_layers.get(gpu_id, []))
        if layers:
            print(f"    GPU {gpu_id}: layers {layers[0]}-{layers[-1]} ({len(layers)} layers)")
        # Also show non-layer components
        non_layer = [k for k, v in device_map.items()
                     if v == gpu_id and not k.startswith("model.layers.")]
        if non_layer:
            print(f"           + {', '.join(non_layer)}")


def benchmark_config(model, tokenizer, batch_size, device_map):
    """Run benchmark and return tokens/sec and per-token latency."""
    # Determine input device (where embeddings live)
    input_device = f"cuda:{device_map['model.embed_tokens']}"

    prompts = ["Explain how pipeline parallelism works in deep learning:"] * batch_size
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True
    ).to(input_device)

    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        with torch.no_grad():
            _ = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id
            )

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(BENCHMARK_ITERATIONS):
        with torch.no_grad():
            _ = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id
            )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_tokens = batch_size * MAX_NEW_TOKENS * BENCHMARK_ITERATIONS
    tokens_per_sec = total_tokens / elapsed
    per_token_ms = (elapsed / (BENCHMARK_ITERATIONS * MAX_NEW_TOKENS)) * 1000
    avg_batch_time = elapsed / BENCHMARK_ITERATIONS

    return tokens_per_sec, per_token_ms, avg_batch_time


def cleanup():
    """Aggressively free GPU memory."""
    gc.collect()
    for i in range(torch.cuda.device_count()):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(i)
    time.sleep(2)


def main():
    num_gpus = torch.cuda.device_count()

    print("=" * 70)
    print("WEEK 3 - EXPERIMENT 3: PIPELINE PARALLELISM PERFORMANCE")
    print("=" * 70)
    print(f"\nModel: {MODEL_NAME}")
    print(f"GPUs available: {num_gpus}")
    print(f"Generation: {MAX_NEW_TOKENS} tokens, {BENCHMARK_ITERATIONS} iterations")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    batch_sizes = [1, 8]

    # Define pipeline configurations to test
    configs = [
        ("1 GPU (baseline)", [0]),
        ("2 GPU pipeline (GPU 0+1)", [0, 1]),
        ("4 GPU pipeline (all)", [0, 1, 2, 3]),
    ]

    # Store all results for final summary
    all_results = {}

    for config_name, gpu_ids in configs:
        print(f"\n{'=' * 70}")
        print(f"CONFIGURATION: {config_name}")
        print(f"{'=' * 70}")

        # Build device map
        device_map, num_layers = build_device_map(MODEL_NAME, len(gpu_ids), gpu_ids)
        print_device_map_summary(device_map, num_layers, gpu_ids)

        # Load model with explicit device map
        print(f"\n  Loading model...")
        start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map=device_map,
            low_cpu_mem_usage=True
        )
        load_time = time.perf_counter() - start
        print(f"  Loaded in {load_time:.1f}s")

        # Show VRAM distribution
        print(f"\n  VRAM usage:")
        for gpu_id in gpu_ids:
            allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
            print(f"    GPU {gpu_id}: {allocated:.2f} GB")

        # Benchmark at each batch size
        for batch_size in batch_sizes:
            print(f"\n  Benchmarking batch_size={batch_size}...")
            tokens_per_sec, per_token_ms, avg_batch_time = benchmark_config(
                model, tokenizer, batch_size, device_map
            )
            per_sample_tps = tokens_per_sec / batch_size

            result_key = (config_name, batch_size)
            all_results[result_key] = {
                'tokens_per_sec': tokens_per_sec,
                'per_sample_tps': per_sample_tps,
                'per_token_ms': per_token_ms,
                'avg_batch_time': avg_batch_time,
                'num_gpus': len(gpu_ids)
            }

            print(f"    Total throughput:  {tokens_per_sec:.1f} tok/s")
            print(f"    Per-sample:        {per_sample_tps:.1f} tok/s")
            print(f"    Per-token latency: {per_token_ms:.2f} ms")
            print(f"    Avg batch time:    {avg_batch_time:.3f}s")

        # Cleanup
        del model
        cleanup()

    # =========================================================
    # COMPARISON TABLES
    # =========================================================
    for batch_size in batch_sizes:
        print(f"\n{'=' * 70}")
        print(f"COMPARISON: batch_size={batch_size}")
        print(f"{'=' * 70}")

        baseline_key = ("1 GPU (baseline)", batch_size)
        baseline_tps = all_results[baseline_key]['tokens_per_sec']
        baseline_latency = all_results[baseline_key]['per_token_ms']

        print(f"\n  {'Configuration':<30} {'Throughput':>10} {'vs Base':>8} "
              f"{'Latency/tok':>12} {'Overhead':>10}")
        print(f"  {'-'*30} {'-'*10} {'-'*8} {'-'*12} {'-'*10}")

        for config_name, gpu_ids in configs:
            key = (config_name, batch_size)
            r = all_results[key]
            speedup = r['tokens_per_sec'] / baseline_tps
            overhead_ms = r['per_token_ms'] - baseline_latency

            print(f"  {config_name:<30} {r['tokens_per_sec']:>8.1f}   "
                  f"{speedup:>6.2f}x  {r['per_token_ms']:>9.2f} ms  "
                  f"{overhead_ms:>+8.2f} ms")

    # =========================================================
    # ANALYSIS
    # =========================================================
    print(f"\n{'=' * 70}")
    print("PIPELINE PARALLELISM OVERHEAD ANALYSIS")
    print(f"{'=' * 70}")

    # Calculate overhead per pipeline stage
    for batch_size in batch_sizes:
        baseline = all_results[("1 GPU (baseline)", batch_size)]
        pipe2 = all_results[("2 GPU pipeline (GPU 0+1)", batch_size)]
        pipe4 = all_results[("4 GPU pipeline (all)", batch_size)]

        overhead_2gpu = pipe2['per_token_ms'] - baseline['per_token_ms']
        overhead_4gpu = pipe4['per_token_ms'] - baseline['per_token_ms']

        # 2-GPU pipeline has 1 stage boundary, 4-GPU has 3
        per_boundary_2gpu = overhead_2gpu / 1
        per_boundary_4gpu = overhead_4gpu / 3

        print(f"\n  Batch size = {batch_size}:")
        print(f"    2-GPU overhead: {overhead_2gpu:.2f} ms total "
              f"({per_boundary_2gpu:.2f} ms per stage boundary)")
        print(f"    4-GPU overhead: {overhead_4gpu:.2f} ms total "
              f"({per_boundary_4gpu:.2f} ms per stage boundary)")
        print(f"    Throughput loss: "
              f"2-GPU = {(1 - pipe2['tokens_per_sec']/baseline['tokens_per_sec'])*100:.1f}%, "
              f"4-GPU = {(1 - pipe4['tokens_per_sec']/baseline['tokens_per_sec'])*100:.1f}%")

    print(f"\n{'=' * 70}")
    print("KEY TAKEAWAYS")
    print(f"{'=' * 70}")
    print(f"""
  1. Pipeline parallelism adds latency per stage boundary due to
     activation transfers over PCIe.

  2. On this system (PCIe x1 for GPUs 1-3), the overhead is
     substantial because activation tensors (~32 MB for hidden
     states) must traverse 1 GB/s links.

  3. Pipeline parallelism is only justified when the model
     CANNOT FIT on a single GPU. For models that fit, single-GPU
     inference is always faster.

  4. When a model must be split, minimize stage boundaries:
     2-GPU split > 4-GPU split (fewer transfers).

  5. Data parallelism (Experiment 2) is strictly better for
     models that fit on one GPU — no communication overhead.
""")


if __name__ == "__main__":
    main()
