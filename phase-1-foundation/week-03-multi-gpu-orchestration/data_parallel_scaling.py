"""
Week 3 - Experiment 2: Data Parallelism Throughput Scaling
Measures throughput scaling as we add independent model replicas across GPUs.
Validates that PCIe x1 bandwidth doesn't affect on-chip inference performance.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import threading
import queue

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MAX_NEW_TOKENS = 50
WARMUP_ITERATIONS = 3
BENCHMARK_ITERATIONS = 10
BATCH_SIZE = 1  # Single request per GPU (real-time chat scenario)

# Also test batched scenario
BATCH_SIZES_TO_TEST = [1, 8, 32]


def load_model(device_id):
    """Load model onto a specific GPU."""
    print(f"  Loading model onto GPU {device_id}...")
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=f"cuda:{device_id}"
    )
    elapsed = time.perf_counter() - start
    print(f"  GPU {device_id}: Model loaded in {elapsed:.1f}s")
    return model


def benchmark_single_gpu(model, tokenizer, device_id, batch_size, iterations):
    """Benchmark inference on a single GPU. Returns tokens/sec."""
    device = f"cuda:{device_id}"
    prompts = ["Explain how GPU memory bandwidth affects inference performance:"] * batch_size
    
    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.pad_token_id)
    
    # Benchmark
    torch.cuda.synchronize(device_id)
    start = time.perf_counter()
    for _ in range(iterations):
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize(device_id)
    elapsed = time.perf_counter() - start
    
    total_tokens = batch_size * MAX_NEW_TOKENS * iterations
    tokens_per_sec = total_tokens / elapsed
    return tokens_per_sec, elapsed


def benchmark_gpu_worker(model, tokenizer, device_id, batch_size, iterations, result_queue):
    """Worker function for threaded multi-GPU benchmark."""
    try:
        tokens_per_sec, elapsed = benchmark_single_gpu(
            model, tokenizer, device_id, batch_size, iterations
        )
        result_queue.put((device_id, tokens_per_sec, elapsed, None))
    except Exception as e:
        result_queue.put((device_id, 0, 0, str(e)))


def main():
    num_gpus = torch.cuda.device_count()
    
    print("=" * 70)
    print("WEEK 3 - EXPERIMENT 2: DATA PARALLELISM THROUGHPUT SCALING")
    print("=" * 70)
    print(f"\nModel: {MODEL_NAME}")
    print(f"GPUs available: {num_gpus}")
    print(f"Generation length: {MAX_NEW_TOKENS} tokens")
    print(f"Benchmark iterations: {BENCHMARK_ITERATIONS}")
    
    # Load tokenizer once (shared, CPU-side)
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load models onto all GPUs
    print(f"\n{'=' * 70}")
    print("PHASE 1: LOADING MODELS (one-time cost)")
    print(f"{'=' * 70}")
    models = {}
    for gpu_id in range(num_gpus):
        models[gpu_id] = load_model(gpu_id)
    
    # Report VRAM usage after loading
    print(f"\nVRAM usage after loading:")
    for gpu_id in range(num_gpus):
        allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
        print(f"  GPU {gpu_id}: {allocated:.2f} GB")
    
    # =========================================================
    # TEST 1: Individual GPU performance (detect x1 vs x16 impact)
    # =========================================================
    print(f"\n{'=' * 70}")
    print("PHASE 2: INDIVIDUAL GPU PERFORMANCE (batch_size=1)")
    print("Purpose: Verify x1 PCIe GPUs match x16 GPU for inference")
    print(f"{'=' * 70}")
    
    individual_results = {}
    for gpu_id in range(num_gpus):
        tokens_per_sec, elapsed = benchmark_single_gpu(
            models[gpu_id], tokenizer, gpu_id, 
            batch_size=1, iterations=BENCHMARK_ITERATIONS
        )
        individual_results[gpu_id] = tokens_per_sec
        pcie_note = "x16" if gpu_id == 0 else "x1"
        print(f"  GPU {gpu_id} ({pcie_note}): {tokens_per_sec:.1f} tok/s")
    
    gpu0_baseline = individual_results[0]
    print(f"\n  Comparison to GPU 0 (x16):")
    for gpu_id in range(1, num_gpus):
        ratio = individual_results[gpu_id] / gpu0_baseline
        print(f"  GPU {gpu_id} (x1): {ratio:.3f}x of GPU 0 — {'✅ No penalty' if ratio > 0.95 else '⚠️ Degraded'}")
    
    # =========================================================
    # TEST 2: Scaling with number of GPUs (data parallelism)
    # =========================================================
    for batch_size in BATCH_SIZES_TO_TEST:
        print(f"\n{'=' * 70}")
        print(f"PHASE 3: DATA PARALLEL SCALING (batch_size={batch_size} per GPU)")
        print(f"{'=' * 70}")
        
        # First, get single-GPU baseline at this batch size
        baseline_tps, _ = benchmark_single_gpu(
            models[0], tokenizer, 0, batch_size, BENCHMARK_ITERATIONS
        )
        print(f"\n  Single GPU baseline: {baseline_tps:.1f} tok/s")
        
        scaling_results = {1: baseline_tps}
        
        for num_active_gpus in [2, 3, 4]:
            if num_active_gpus > num_gpus:
                break
            
            print(f"\n  Testing {num_active_gpus} GPUs in parallel...")
            
            result_queue = queue.Queue()
            threads = []
            
            for gpu_id in range(num_active_gpus):
                t = threading.Thread(
                    target=benchmark_gpu_worker,
                    args=(models[gpu_id], tokenizer, gpu_id, batch_size,
                          BENCHMARK_ITERATIONS, result_queue)
                )
                threads.append(t)
            
            # Start all threads simultaneously
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            # Collect results
            gpu_results = {}
            total_tps = 0
            while not result_queue.empty():
                gpu_id, tps, elapsed, error = result_queue.get()
                if error:
                    print(f"    GPU {gpu_id}: ERROR — {error}")
                else:
                    gpu_results[gpu_id] = tps
                    total_tps += tps
                    print(f"    GPU {gpu_id}: {tps:.1f} tok/s")
            
            scaling_results[num_active_gpus] = total_tps
            
            ideal = baseline_tps * num_active_gpus
            efficiency = (total_tps / ideal) * 100
            print(f"    ─────────────────────────")
            print(f"    Total: {total_tps:.1f} tok/s")
            print(f"    Ideal: {ideal:.1f} tok/s ({num_active_gpus}x baseline)")
            print(f"    Efficiency: {efficiency:.1f}%")
        
        # Summary table for this batch size
        print(f"\n  ┌─────────────────────────────────────────────────────┐")
        print(f"  │ SCALING SUMMARY — batch_size={batch_size} per GPU{' ' * (22 - len(str(batch_size)))}│")
        print(f"  ├────────┬───────────┬───────────┬──────────┬─────────┤")
        print(f"  │  GPUs  │ Total TPS │ Ideal TPS │  Scale   │  Eff.   │")
        print(f"  ├────────┼───────────┼───────────┼──────────┼─────────┤")
        for n_gpus, total_tps in sorted(scaling_results.items()):
            ideal = baseline_tps * n_gpus
            scale = total_tps / baseline_tps
            eff = (total_tps / ideal) * 100
            print(f"  │  {n_gpus}x    │ {total_tps:>7.0f}   │ {ideal:>7.0f}   │  {scale:.2f}x   │ {eff:5.1f}%  │")
        print(f"  └────────┴───────────┴───────────┴──────────┴─────────┘")
    
    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    print(f"\n{'=' * 70}")
    print("KEY FINDINGS")
    print(f"{'=' * 70}")
    print(f"\n  1. PCIe x1 vs x16 inference impact:")
    for gpu_id in range(1, num_gpus):
        ratio = individual_results[gpu_id] / gpu0_baseline
        print(f"     GPU {gpu_id} (x1): {ratio:.1%} of GPU 0 (x16) performance")
    print(f"\n  2. Data parallel scaling across batch sizes tested:")
    print(f"     (see tables above)")
    print(f"\n  3. Production implication:")
    print(f"     4x RTX 3090 with data parallelism = 4 independent inference servers")
    print(f"     Total system throughput ≈ 4 × single GPU throughput")
    print(f"     No inter-GPU communication needed during inference")


if __name__ == "__main__":
    main()
