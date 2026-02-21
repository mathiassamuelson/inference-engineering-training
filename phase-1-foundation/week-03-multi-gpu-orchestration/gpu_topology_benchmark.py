"""
Week 3 - Experiment 1: GPU Topology & Communication Baseline
Measures inter-GPU bandwidth and latency across all 4x RTX 3090s.
"""
import torch
import time
import itertools

def measure_bandwidth(src_device, dst_device, size_mb=256, iterations=20):
    """Measure unidirectional transfer bandwidth between two devices."""
    size_bytes = size_mb * 1024 * 1024
    num_elements = size_bytes // 4  # float32

    src_tensor = torch.randn(num_elements, device=src_device)

    # Warmup
    for _ in range(5):
        dst_tensor = src_tensor.to(dst_device)
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        dst_tensor = src_tensor.to(dst_device)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    bandwidth_gbps = (size_mb * iterations / 1024) / elapsed
    avg_latency_ms = (elapsed / iterations) * 1000
    return bandwidth_gbps, avg_latency_ms


def measure_latency(src_device, dst_device, iterations=100):
    """Measure latency for small transfers (simulates all-reduce overhead)."""
    # Small tensor: 4KB (typical gradient slice)
    small_tensor = torch.randn(1024, device=src_device)

    # Warmup
    for _ in range(10):
        _ = small_tensor.to(dst_device)
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = small_tensor.to(dst_device)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (elapsed / iterations) * 1000  # ms


def main():
    num_gpus = torch.cuda.device_count()
    print("=" * 70)
    print("WEEK 3 - EXPERIMENT 1: GPU TOPOLOGY & COMMUNICATION BASELINE")
    print("=" * 70)

    # Section 1: GPU Info
    print(f"\nDetected {num_gpus} GPUs:\n")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")
        print(f"         VRAM: {props.total_memory / 1e9:.1f} GB")
        print(f"         SMs: {props.multi_processor_count}")
        print(f"         Compute: {props.major}.{props.minor}")

    # Section 2: Peer Access
    print(f"\n{'=' * 70}")
    print("PEER ACCESS MATRIX (can GPUs transfer directly?)")
    print(f"{'=' * 70}\n")
    header = "      " + "".join(f"  GPU {j}  " for j in range(num_gpus))
    print(header)
    print("      " + "-" * (9 * num_gpus))
    for i in range(num_gpus):
        row = f"GPU {i} |"
        for j in range(num_gpus):
            if i == j:
                row += "    -    "
            else:
                can_access = torch.cuda.can_device_access_peer(i, j)
                row += f"  {'YES':^5}  " if can_access else f"  {'NO':^5}  "
        print(row)

    # Section 3: Large Transfer Bandwidth (all pairs)
    print(f"\n{'=' * 70}")
    print("BANDWIDTH: LARGE TRANSFERS (256 MB, simulates weight/activation transfers)")
    print(f"{'=' * 70}\n")
    print(f"{'Source':>8} → {'Dest':<8}  {'Bandwidth':>10}  {'Latency':>10}")
    print(f"{'-' * 8}   {'-' * 8}  {'-' * 10}  {'-' * 10}")

    bandwidth_matrix = {}
    for i, j in itertools.permutations(range(num_gpus), 2):
        src = f"cuda:{i}"
        dst = f"cuda:{j}"
        bw, lat = measure_bandwidth(src, dst)
        bandwidth_matrix[(i, j)] = bw
        print(f"  GPU {i}  →  GPU {j}    {bw:7.2f} GB/s  {lat:7.2f} ms")

    # Host transfers
    print(f"\n{'Source':>8} → {'Dest':<8}  {'Bandwidth':>10}  {'Latency':>10}")
    print(f"{'-' * 8}   {'-' * 8}  {'-' * 10}  {'-' * 10}")
    for i in range(num_gpus):
        # Host to Device
        src_cpu = torch.randn(256 * 1024 * 1024 // 4)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            _ = src_cpu.to(f"cuda:{i}")
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        h2d_bw = (256 * 10 / 1024) / elapsed
        print(f"  Host  →  GPU {i}    {h2d_bw:7.2f} GB/s")

        # Device to Host
        src_gpu = torch.randn(256 * 1024 * 1024 // 4, device=f"cuda:{i}")
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            _ = src_gpu.cpu()
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        d2h_bw = (256 * 10 / 1024) / elapsed
        print(f"  GPU {i} →  Host    {d2h_bw:7.2f} GB/s")

    # Section 4: Small Transfer Latency
    print(f"\n{'=' * 70}")
    print("LATENCY: SMALL TRANSFERS (4 KB, simulates all-reduce overhead)")
    print(f"{'=' * 70}\n")
    print(f"{'Source':>8} → {'Dest':<8}  {'Latency':>10}")
    print(f"{'-' * 8}   {'-' * 8}  {'-' * 10}")

    for i, j in itertools.permutations(range(num_gpus), 2):
        lat = measure_latency(f"cuda:{i}", f"cuda:{j}")
        print(f"  GPU {i}  →  GPU {j}    {lat:7.3f} ms")

    # Section 5: Simulated All-Reduce
    print(f"\n{'=' * 70}")
    print("SIMULATED ALL-REDUCE (ring pattern, 32 MB — typical activation size)")
    print(f"{'=' * 70}\n")

    size_mb = 32
    num_elements = size_mb * 1024 * 1024 // 4
    tensors = [torch.randn(num_elements, device=f"cuda:{i}") for i in range(num_gpus)]

    # Warmup
    for _ in range(3):
        for step in range(num_gpus - 1):
            for i in range(num_gpus):
                src = i
                dst = (i + 1) % num_gpus
                tensors[dst] += tensors[src].to(f"cuda:{dst}")
            torch.cuda.synchronize()

    # Reset tensors
    tensors = [torch.randn(num_elements, device=f"cuda:{i}") for i in range(num_gpus)]

    # Benchmark ring all-reduce
    iterations = 10
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        for step in range(num_gpus - 1):
            for i in range(num_gpus):
                src = i
                dst = (i + 1) % num_gpus
                tensors[dst] += tensors[src].to(f"cuda:{dst}")
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    all_reduce_ms = (elapsed / iterations) * 1000
    # Effective bandwidth: each GPU sends (N-1)/N of data in ring
    effective_data_gb = size_mb * (num_gpus - 1) / num_gpus / 1024
    effective_bw = effective_data_gb / (elapsed / iterations)

    print(f"  Ring all-reduce time: {all_reduce_ms:.2f} ms")
    print(f"  Effective bandwidth: {effective_bw:.2f} GB/s")
    print(f"  Note: NVLink systems achieve 5-10x higher bandwidth here")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY & IMPLICATIONS FOR MULTI-GPU STRATEGIES")
    print(f"{'=' * 70}")

    avg_gpu_bw = sum(bandwidth_matrix.values()) / len(bandwidth_matrix)
    print(f"\n  Average GPU-to-GPU bandwidth: {avg_gpu_bw:.2f} GB/s")
    print(f"  PCIe 3.0 x16 theoretical max: ~15.75 GB/s per direction")
    print(f"  PCIe 4.0 x16 theoretical max: ~31.5 GB/s per direction")
    print(f"  NVLink (A100) for comparison:  ~600 GB/s bidirectional")

    print(f"\n  Implications:")
    print(f"  - Tensor Parallelism: Each token requires all-reduce of activations")
    print(f"    At {all_reduce_ms:.1f} ms per all-reduce, adds ~{all_reduce_ms:.0f} ms per layer")
    print(f"    For 80-layer model: ~{all_reduce_ms * 80:.0f} ms overhead per token")
    print(f"  - Pipeline Parallelism: Only transfers activations between stages")
    print(f"    Much less frequent communication, better for PCIe")
    print(f"  - Data Parallelism: No inter-GPU communication during inference")
    print(f"    Best throughput scaling on PCIe systems")


if __name__ == "__main__":
    main()
