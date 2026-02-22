"""
Week 3 - Experiment 4: CUDA Streams & Async Execution
Demonstrates how CUDA streams enable overlapping operations and why
this matters for production inference performance.
"""
import torch
import time
import threading

WARMUP = 3
ITERATIONS = 20


def gpu_compute_work(device, size=4096, iterations=200):
    """Simulate meaningful GPU compute (matrix multiplications)."""
    a = torch.randn(size, size, device=device, dtype=torch.float16)
    b = torch.randn(size, size, device=device, dtype=torch.float16)
    for _ in range(iterations):
        c = torch.mm(a, b)
    return c


def measure_time(func, sync_device=None):
    """Measure wall-clock time for a function, with optional CUDA sync."""
    if sync_device is not None:
        torch.cuda.synchronize(sync_device)
    start = time.perf_counter()
    result = func()
    if sync_device is not None:
        torch.cuda.synchronize(sync_device)
    elapsed = time.perf_counter() - start
    return elapsed, result


# ==========================================================================
# EXPERIMENT 4A: Default Stream Serialization
# ==========================================================================
def experiment_4a():
    """
    Show that operations on the DEFAULT stream are serialized.
    Two independent matrix multiplications run back-to-back, not overlapped.
    """
    print("=" * 70)
    print("EXPERIMENT 4A: DEFAULT STREAM SERIALIZATION")
    print("=" * 70)
    print("\nAll operations on a GPU's default stream execute in order.")
    print("Even independent operations cannot overlap.\n")

    device = "cuda:0"
    size = 4096
    compute_iters = 200

    # Warmup
    gpu_compute_work(device, size, 10)
    torch.cuda.synchronize()

    # Measure single workload
    torch.cuda.synchronize()
    start = time.perf_counter()
    gpu_compute_work(device, size, compute_iters)
    torch.cuda.synchronize()
    single_time = time.perf_counter() - start

    # Measure two sequential workloads on default stream
    torch.cuda.synchronize()
    start = time.perf_counter()
    gpu_compute_work(device, size, compute_iters)
    gpu_compute_work(device, size, compute_iters)
    torch.cuda.synchronize()
    serial_time = time.perf_counter() - start

    print(f"  Single workload:     {single_time * 1000:.1f} ms")
    print(f"  Two serial (1 stream): {serial_time * 1000:.1f} ms")
    print(f"  Expected (2x single):  {single_time * 2 * 1000:.1f} ms")
    print(f"  Ratio: {serial_time / single_time:.2f}x (expect ~2.0x)")
    print(f"\n  → Default stream serializes everything: two independent")
    print(f"    workloads take 2x as long, no overlap possible.")

    return single_time


# ==========================================================================
# EXPERIMENT 4B: Multiple Streams on Same GPU
# ==========================================================================
def experiment_4b(single_time):
    """
    Show that multiple streams on the SAME GPU can overlap operations,
    but only if the GPU has enough resources (SMs) to run both.
    """
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 4B: MULTIPLE STREAMS ON SAME GPU")
    print("=" * 70)
    print("\nMultiple streams allow the GPU scheduler to interleave operations.")
    print("Overlap depends on available SM resources.\n")

    device = "cuda:0"
    size = 4096
    compute_iters = 200

    # Warmup
    stream1 = torch.cuda.Stream(device=device)
    stream2 = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream1):
        gpu_compute_work(device, size, 10)
    with torch.cuda.stream(stream2):
        gpu_compute_work(device, size, 10)
    torch.cuda.synchronize()

    # Launch two workloads on separate streams
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.cuda.stream(stream1):
        gpu_compute_work(device, size, compute_iters)
    with torch.cuda.stream(stream2):
        gpu_compute_work(device, size, compute_iters)

    torch.cuda.synchronize()
    two_stream_time = time.perf_counter() - start

    # Compare: if workloads are large enough to saturate SMs,
    # overlap may be minimal
    overlap_pct = (1 - (two_stream_time / (single_time * 2))) * 100

    print(f"  Single workload:         {single_time * 1000:.1f} ms")
    print(f"  Two streams (same GPU):  {two_stream_time * 1000:.1f} ms")
    print(f"  Ideal (full overlap):    {single_time * 1000:.1f} ms")
    print(f"  Serial (no overlap):     {single_time * 2 * 1000:.1f} ms")
    print(f"  Actual overlap: {overlap_pct:.1f}%")

    if two_stream_time > single_time * 1.8:
        print(f"\n  → Minimal overlap. Large matmuls saturate all 82 SMs,")
        print(f"    leaving no room for the second stream to execute.")
        print(f"    Streams help when workloads DON'T fully utilize the GPU.")
    else:
        print(f"\n  → Some overlap achieved. The GPU scheduler interleaved")
        print(f"    operations from both streams across available SMs.")

    return two_stream_time


# ==========================================================================
# EXPERIMENT 4C: Overlapping Compute and Data Transfer
# ==========================================================================
def experiment_4c():
    """
    The most important async pattern: overlap data transfers with computation.
    This is exactly what production frameworks do — while one batch computes,
    the next batch's data is being transferred.
    """
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 4C: OVERLAPPING COMPUTE AND DATA TRANSFER")
    print("=" * 70)
    print("\nThe key production pattern: transfer data while GPU computes.")
    print("Requires pinned (page-locked) memory for async transfers.\n")

    device = "cuda:0"
    size = 4096
    compute_iters = 200
    transfer_size_mb = 128
    transfer_elements = transfer_size_mb * 1024 * 1024 // 2  # FP16

    # Create pinned memory (required for async H2D transfer)
    cpu_tensor_pinned = torch.randn(transfer_elements, dtype=torch.float16).pin_memory()
    cpu_tensor_paged = torch.randn(transfer_elements, dtype=torch.float16)

    # Warmup
    gpu_compute_work(device, size, 10)
    _ = cpu_tensor_pinned.to(device, non_blocking=True)
    torch.cuda.synchronize()

    # Measure compute alone
    torch.cuda.synchronize()
    start = time.perf_counter()
    gpu_compute_work(device, size, compute_iters)
    torch.cuda.synchronize()
    compute_time = time.perf_counter() - start

    # Measure transfer alone (pinned)
    torch.cuda.synchronize()
    start = time.perf_counter()
    _ = cpu_tensor_pinned.to(device, non_blocking=False)
    torch.cuda.synchronize()
    transfer_time_pinned = time.perf_counter() - start

    # Measure transfer alone (paged/regular)
    torch.cuda.synchronize()
    start = time.perf_counter()
    _ = cpu_tensor_paged.to(device, non_blocking=False)
    torch.cuda.synchronize()
    transfer_time_paged = time.perf_counter() - start

    # Serial: compute then transfer
    torch.cuda.synchronize()
    start = time.perf_counter()
    gpu_compute_work(device, size, compute_iters)
    _ = cpu_tensor_pinned.to(device, non_blocking=False)
    torch.cuda.synchronize()
    serial_time = time.perf_counter() - start

    # Overlapped: compute on default stream, transfer on separate stream
    transfer_stream = torch.cuda.Stream(device=device)
    torch.cuda.synchronize()
    start = time.perf_counter()

    # Launch compute on default stream
    gpu_compute_work(device, size, compute_iters)

    # Simultaneously launch transfer on separate stream
    with torch.cuda.stream(transfer_stream):
        gpu_tensor = cpu_tensor_pinned.to(device, non_blocking=True)

    # Wait for both
    torch.cuda.synchronize()
    overlapped_time = time.perf_counter() - start

    savings_pct = (1 - overlapped_time / serial_time) * 100
    overlap_achieved = serial_time - overlapped_time

    print(f"  Transfer details:")
    print(f"    Size: {transfer_size_mb} MB (FP16)")
    print(f"    Pinned memory transfer:  {transfer_time_pinned * 1000:.1f} ms")
    print(f"    Regular memory transfer: {transfer_time_paged * 1000:.1f} ms")
    print(f"    Pinned speedup: {transfer_time_paged / transfer_time_pinned:.2f}x")

    print(f"\n  Timing comparison:")
    print(f"    Compute only:    {compute_time * 1000:.1f} ms")
    print(f"    Transfer only:   {transfer_time_pinned * 1000:.1f} ms")
    print(f"    Serial (both):   {serial_time * 1000:.1f} ms")
    print(f"    Overlapped:      {overlapped_time * 1000:.1f} ms")
    print(f"    Time saved:      {overlap_achieved * 1000:.1f} ms ({savings_pct:.1f}%)")

    print(f"\n  → Overlapping hides the transfer behind compute time.")
    print(f"    This is how vLLM prefetches the next batch while")
    print(f"    the current batch is being processed.")


# ==========================================================================
# EXPERIMENT 4D: Multi-GPU Async Independence
# ==========================================================================
def experiment_4d():
    """
    Demonstrate that operations on different GPUs are naturally async.
    Each GPU has its own default stream — no explicit stream management needed.
    This is why data parallelism 'just works'.
    """
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print("\n  Skipping 4D: need 2+ GPUs")
        return

    print(f"\n{'=' * 70}")
    print("EXPERIMENT 4D: MULTI-GPU ASYNC INDEPENDENCE")
    print("=" * 70)
    print("\nDifferent GPUs have independent default streams.")
    print("Operations on separate GPUs run concurrently by default.\n")

    size = 4096
    compute_iters = 200

    # Warmup all GPUs
    for i in range(num_gpus):
        gpu_compute_work(f"cuda:{i}", size, 10)
    torch.cuda.synchronize()

    # Measure single GPU
    torch.cuda.synchronize()
    start = time.perf_counter()
    gpu_compute_work("cuda:0", size, compute_iters)
    torch.cuda.synchronize()
    single_gpu_time = time.perf_counter() - start

    # Measure all GPUs serial (one after another, synchronizing between)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(num_gpus):
        gpu_compute_work(f"cuda:{i}", size, compute_iters)
        torch.cuda.synchronize(i)
    serial_all_time = time.perf_counter() - start

    # Measure all GPUs concurrent (launch all, then sync all)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(num_gpus):
        gpu_compute_work(f"cuda:{i}", size, compute_iters)
    # All launches are non-blocking from CPU perspective
    # Now sync all
    for i in range(num_gpus):
        torch.cuda.synchronize(i)
    concurrent_time = time.perf_counter() - start

    # Measure using threads (explicit parallelism for CPU-bound orchestration)
    def gpu_work_thread(device_id, results):
        gpu_compute_work(f"cuda:{device_id}", size, compute_iters)
        torch.cuda.synchronize(device_id)
        results[device_id] = True

    thread_results = {}
    torch.cuda.synchronize()
    start = time.perf_counter()
    threads = []
    for i in range(num_gpus):
        t = threading.Thread(target=gpu_work_thread, args=(i, thread_results))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    threaded_time = time.perf_counter() - start

    ideal_concurrent = single_gpu_time  # Perfect overlap = single GPU time

    print(f"  Single GPU compute:      {single_gpu_time * 1000:.1f} ms")
    print(f"  {num_gpus} GPUs serial:          {serial_all_time * 1000:.1f} ms "
          f"({serial_all_time / single_gpu_time:.2f}x single)")
    print(f"  {num_gpus} GPUs concurrent:      {concurrent_time * 1000:.1f} ms "
          f"({concurrent_time / single_gpu_time:.2f}x single)")
    print(f"  {num_gpus} GPUs threaded:        {threaded_time * 1000:.1f} ms "
          f"({threaded_time / single_gpu_time:.2f}x single)")
    print(f"  Ideal ({num_gpus}-way overlap):   {ideal_concurrent * 1000:.1f} ms")

    concurrent_efficiency = (single_gpu_time / concurrent_time) * 100
    threaded_efficiency = (single_gpu_time / threaded_time) * 100

    print(f"\n  Concurrent efficiency: {concurrent_efficiency:.1f}%")
    print(f"  Threaded efficiency:   {threaded_efficiency:.1f}%")

    print(f"\n  → GPU operations on different devices are inherently async.")
    print(f"    The CUDA runtime launches kernels without waiting.")
    print(f"    Python threads help when CPU-side work (tokenization,")
    print(f"    post-processing) would otherwise serialize the launches.")


# ==========================================================================
# EXPERIMENT 4E: Synchronization Cost
# ==========================================================================
def experiment_4e():
    """
    Measure the cost of different synchronization methods.
    Over-synchronizing is a common performance mistake.
    """
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 4E: SYNCHRONIZATION COST")
    print("=" * 70)
    print("\nExcessive synchronization kills async benefits.")
    print("Each sync forces CPU to wait for GPU to finish.\n")

    device = "cuda:0"
    size = 2048
    iters_per_call = 50

    # Warmup
    gpu_compute_work(device, size, 10)
    torch.cuda.synchronize()

    # Pattern 1: Single sync at end (correct for benchmarking)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        gpu_compute_work(device, size, iters_per_call)
    torch.cuda.synchronize()
    single_sync_time = time.perf_counter() - start

    # Pattern 2: Sync after every operation (over-synchronizing)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        gpu_compute_work(device, size, iters_per_call)
        torch.cuda.synchronize()
    over_sync_time = time.perf_counter() - start

    # Pattern 3: CUDA events (lightweight synchronization)
    events = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        gpu_compute_work(device, size, iters_per_call)
        event = torch.cuda.Event()
        event.record()
        events.append(event)
    # Wait for last event only
    events[-1].synchronize()
    event_sync_time = time.perf_counter() - start

    # Measure bare sync cost (GPU idle)
    torch.cuda.synchronize()
    sync_times = []
    for _ in range(100):
        start = time.perf_counter()
        torch.cuda.synchronize()
        sync_times.append((time.perf_counter() - start) * 1000)
    avg_sync_ms = sum(sync_times) / len(sync_times)

    overhead_pct = ((over_sync_time - single_sync_time) / single_sync_time) * 100

    print(f"  Bare synchronize() cost: {avg_sync_ms:.3f} ms (GPU idle)")
    print(f"")
    print(f"  10 workloads, different sync patterns:")
    print(f"    Single sync at end:     {single_sync_time * 1000:.1f} ms")
    print(f"    Sync after each:        {over_sync_time * 1000:.1f} ms")
    print(f"    CUDA events:            {event_sync_time * 1000:.1f} ms")
    print(f"    Over-sync overhead:     {overhead_pct:.1f}%")

    print(f"\n  → Over-synchronizing forces the CPU to stall, preventing it")
    print(f"    from queuing the next operation. Pipeline frameworks avoid")
    print(f"    this by using CUDA events for fine-grained dependencies")
    print(f"    instead of full device synchronization.")


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    print("=" * 70)
    print("WEEK 3 - EXPERIMENT 4: CUDA STREAMS & ASYNC EXECUTION")
    print("=" * 70)
    print(f"\nGPUs: {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}")
    print(f"SMs per GPU: {torch.cuda.get_device_properties(0).multi_processor_count}")

    single_time = experiment_4a()
    experiment_4b(single_time)
    experiment_4c()
    experiment_4d()
    experiment_4e()

    print(f"\n{'=' * 70}")
    print("SUMMARY: CUDA ASYNC EXECUTION MODEL")
    print(f"{'=' * 70}")
    print(f"""
  Core Concepts:
  1. DEFAULT STREAM: All ops on a GPU serialize unless you use streams
  2. MULTIPLE STREAMS: Enable overlap, but large kernels saturate SMs
  3. COMPUTE + TRANSFER OVERLAP: The key production optimization
     - Requires pinned (page-locked) memory
     - Separate stream for data movement
     - This is how vLLM prefetches next batch during inference
  4. MULTI-GPU INDEPENDENCE: Different GPUs are naturally async
     - No explicit streams needed for data parallelism
     - Python threads help with CPU-side orchestration
  5. SYNC COST: Over-synchronizing kills performance
     - Sync only when you need results
     - Use CUDA events for fine-grained dependencies

  Production Relevance:
  - vLLM uses async prefetching to hide data movement
  - Triton uses stream management for multi-model serving
  - Pipeline parallelism frameworks use events to coordinate stages
  - Understanding streams explains WHY frameworks are faster than
    naive PyTorch implementations
""")


if __name__ == "__main__":
    main()
