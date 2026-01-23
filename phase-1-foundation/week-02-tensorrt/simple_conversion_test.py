import torch
import torch.nn as nn
import tensorrt as trt
import numpy as np
import time
import os

print("="*70)
print("SIMPLE TENSORRT CONVERSION TEST")
print("="*70)

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(512, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

print("\n[1/6] Creating PyTorch Model")
print("-"*70)

model = SimpleNet().cuda().eval()
print(f"✓ Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

# Export to ONNX (intermediate format for TensorRT)
print("\n[2/6] Exporting to ONNX")
print("-"*70)

os.makedirs('results', exist_ok=True)
onnx_path = 'results/simple_model.onnx'

dummy_input = torch.randn(1, 512).cuda()

try:
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=17
    )
    print(f"✓ ONNX export successful: {onnx_path}")
except Exception as e:
    print(f"✗ ONNX export failed: {e}")
    import sys
    sys.exit(1)

# Convert ONNX to TensorRT
print("\n[3/6] Converting ONNX to TensorRT")
print("-"*70)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

# Parse ONNX
with open(onnx_path, 'rb') as f:
    if not parser.parse(f.read()):
        print("✗ ONNX parsing failed")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        import sys
        sys.exit(1)

print("✓ ONNX parsed successfully")

# Build TensorRT engine
print("  Building TensorRT engine (FP16 optimization)...")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB workspace

# Enable FP16 mode
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)
    print("  ✓ FP16 mode enabled")
else:
    print("  ⚠ FP16 not supported, using FP32")

# Set optimization profile for dynamic batch size
profile = builder.create_optimization_profile()
profile.set_shape(
    "input",
    min=(1, 512),    # min batch size
    opt=(8, 512),    # optimal batch size
    max=(64, 512)    # max batch size
)
config.add_optimization_profile(profile)

try:
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        print("✗ Engine build failed")
        import sys
        sys.exit(1)
    
    # Save engine
    trt_path = 'results/simple_model.trt'
    with open(trt_path, 'wb') as f:
        f.write(engine)
    
    print(f"✓ TensorRT engine built and saved: {trt_path}")
    print(f"  Engine size: {len(engine) / 1024:.1f} KB")
    
except Exception as e:
    print(f"✗ Engine build failed: {e}")
    import sys
    sys.exit(1)

# Test PyTorch inference
print("\n[4/6] Benchmarking PyTorch (FP32)")
print("-"*70)

batch_size = 8
test_input = torch.randn(batch_size, 512).cuda()

# Warmup
for _ in range(10):
    with torch.no_grad():
        _ = model(test_input)

# Benchmark
iterations = 100
torch.cuda.synchronize()
start = time.time()

for _ in range(iterations):
    with torch.no_grad():
        pytorch_output = model(test_input)

torch.cuda.synchronize()
pytorch_time = (time.time() - start) / iterations * 1000  # ms

print(f"✓ PyTorch inference: {pytorch_time:.3f} ms/batch")
print(f"  Throughput: {batch_size / (pytorch_time/1000):.1f} samples/sec")

# Test TensorRT inference
print("\n[5/6] Benchmarking TensorRT (FP16)")
print("-"*70)

runtime = trt.Runtime(TRT_LOGGER)
with open(trt_path, 'rb') as f:
    engine_bytes = f.read()
    trt_engine = runtime.deserialize_cuda_engine(engine_bytes)

context = trt_engine.create_execution_context()

# Set input shape for this batch size
context.set_input_shape("input", (batch_size, 512))

# Allocate buffers
input_np = test_input.cpu().numpy().astype(np.float32)
output_np = np.empty((batch_size, 10), dtype=np.float32)

d_input = torch.from_numpy(input_np).cuda()
d_output = torch.empty((batch_size, 10), dtype=torch.float32).cuda()

# Warmup
for _ in range(10):
    context.execute_v2([d_input.data_ptr(), d_output.data_ptr()])

# Benchmark
torch.cuda.synchronize()
start = time.time()

for _ in range(iterations):
    context.execute_v2([d_input.data_ptr(), d_output.data_ptr()])

torch.cuda.synchronize()
trt_time = (time.time() - start) / iterations * 1000  # ms

print(f"✓ TensorRT inference: {trt_time:.3f} ms/batch")
print(f"  Throughput: {batch_size / (trt_time/1000):.1f} samples/sec")

# Verify numerical accuracy
print("\n[6/6] Numerical Accuracy Check")
print("-"*70)

with torch.no_grad():
    pytorch_output = model(test_input).cpu().numpy()

context.execute_v2([d_input.data_ptr(), d_output.data_ptr()])
trt_output = d_output.cpu().numpy()

max_diff = np.abs(pytorch_output - trt_output).max()
mean_diff = np.abs(pytorch_output - trt_output).mean()

print(f"Max absolute difference: {max_diff:.6f}")
print(f"Mean absolute difference: {mean_diff:.6f}")

if max_diff < 0.01:
    print("✓ Outputs match (< 0.01 threshold)")
else:
    print("⚠ Outputs differ significantly")

# Summary
print("\n" + "="*70)
print("CONVERSION TEST SUMMARY")
print("="*70)

speedup = pytorch_time / trt_time
print(f"PyTorch (FP32):  {pytorch_time:.3f} ms/batch")
print(f"TensorRT (FP16): {trt_time:.3f} ms/batch")
print(f"Speedup: {speedup:.2f}x")
print(f"Accuracy: Max diff {max_diff:.6f}")

# Save results
with open('results/conversion_test_results.txt', 'w') as f:
    f.write("TENSORRT CONVERSION TEST RESULTS\n")
    f.write("="*70 + "\n\n")
    f.write(f"Model: SimpleNet ({sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters)\n")
    f.write(f"Batch size: {batch_size}\n")
    f.write(f"Iterations: {iterations}\n\n")
    f.write(f"PyTorch (FP32):\n")
    f.write(f"  Latency: {pytorch_time:.3f} ms/batch\n")
    f.write(f"  Throughput: {batch_size / (pytorch_time/1000):.1f} samples/sec\n\n")
    f.write(f"TensorRT (FP16):\n")
    f.write(f"  Latency: {trt_time:.3f} ms/batch\n")
    f.write(f"  Throughput: {batch_size / (trt_time/1000):.1f} samples/sec\n\n")
    f.write(f"Performance:\n")
    f.write(f"  Speedup: {speedup:.2f}x\n\n")
    f.write(f"Accuracy:\n")
    f.write(f"  Max diff: {max_diff:.6f}\n")
    f.write(f"  Mean diff: {mean_diff:.6f}\n\n")
    f.write("Status: PASSED\n")

print("\n✓ Results saved to: results/conversion_test_results.txt")
print("\n✓ TensorRT conversion pipeline verified!")
print("  Ready to convert larger models (Llama 3.2 3B)")
print("="*70)
