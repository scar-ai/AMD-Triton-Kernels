# Making High performance MI300X kernels with Triton.

---

![Language](https://img.shields.io/badge/Language-Python-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Triton%20%7C%20PyTorch-orange.svg)
![Contest](https://img.shields.io/badge/Origin-AMD%20Inference%20Sprint-red.svg)

A collection of high-performance Triton kernels designed to accelerate state-of-the-art deepseek models on GPUs. This repository features an optimized FP8 matrix multiplication kernel for weights scaling following [this taskfrom gpumode](https://www.gpumode.com/leaderboard/399) and a complete, from-scratch implementation of a DeepSeek-style Mixture of Experts (MoE) layer following [this task from gpumode](https://www.gpumode.com/leaderboard/430). These kernels were originally developed for the AMD Inference Sprint.

## Key Features

This repository provides two main, self-contained examples of advanced GPU acceleration with Triton:

### 1. Mixture of Experts (MoE) Layer (`amd-moe.py`)

A complete, end-to-end implementation of a parallelized Mixture of Experts layer using a triton kernel.

- **Expert Parallelism:** Efficiently routes and processes tokens through different experts in parallel.
- **Custom Kernel Suite:** Includes from-scratch Triton implementations for:
  - **`matmul`**: Tiled matrix multiplication with group scheduling for improved L2 cache utilization.
  - **`silu`**: Fused SiLU activation function.
  - **`softmax`**: Stable softmax over the last dimension for router probabilities.
  - **`transpose`**: Optimized 2D matrix transpose.
  - **`logsumexp` & `mean`**: Fused reduction kernels for calculating auxiliary losses.
  - **`gather_rows`**: Parallel row gathering for creating expert-specific mini-batches.
  - **`scatter_add_rows`**: Atomic scatter-add for combining expert outputs without race conditions.
- **Auxiliary Loss Calculation:** Computes `router_z_loss` and `load_balancing_loss` on the GPU for a complete training/inference step.

### 2. FP8 Tiled GEMM (`amd-fp8-gemm.py`)

A fast and flexible kernel for General Matrix Multiplication (GEMM) for FP8 weights scaling.

- **Low-Precision, High-Performance:** Performs `C = A @ B` where `A` and `B` are FP8 tensors, with accumulation in FP32 for precision.
- **Dynamic Scaling:** Handles per-block or per-tensor scaling factors (`a_scale`, `b_scale`) to maintain numerical accuracy.
- **Triton Autotuner:** Includes an extensive autotuning configuration to automatically find the optimal block sizes, number of warps, and stages for any given matrix shape on your specific GPU.
- **Grouped Scheduling:** Uses program grouping to enhance data reuse in the L2 cache, critical for large matrix multiplications.

## What is this for?

- **A Learning Resource:** Unfortunately, GPU kernel writing is a very niche side of computer science and LLM models from across the board don't have enough training data on these subject to help one learn it. Feel free to use this repo to get started in wrinting Triton kernels.
- **Performance-Oriented:** These kernels are designed for speed, using techniques like tiling, memory coalescing, shared memory, and autotuning to squeeze performance out of the hardware (specifically the AMD MI300X on which was the AMD inference sprint based).
- **Beyond the Basics:** Goes beyond simple element-wise kernels to tackle some of the most important and performance-critical operations in modern transformer and mixture-of-experts architectures.

## Getting Started

### Prerequisites

Ensure you have a compatible GPU and the necessary libraries installed.

```bash
pip install torch triton
```

### Usage Example: MoE Layer

The `amd-moe.py` file contains a `custom_kernel` wrapper function that takes the input tensor, weights, and configuration, and returns the processed tensor and auxiliary data.

```python
import torch
from amd_moe import custom_kernel

# 1. Define model and input parameters
bs, seq_len, d_hidden = 2, 1024, 4096
num_experts = 8
top_k = 2
device = 'cuda'

# 2. Create dummy input and configuration
input_tensor = torch.randn(bs, seq_len, d_hidden, device=device, dtype=torch.bfloat16)
config = {
    "n_routed_experts": num_experts,
    "n_experts_per_token": top_k,
    "router_z_loss_coef": 0.001,
    "load_balance_loss_coef": 0.01,
}

# 3. Create dummy weights
weights = {}
# --- Router weights
weights['router.weight'] = torch.randn(num_experts, d_hidden, device=device, dtype=torch.bfloat16)
# --- Shared expert weights
weights['shared_experts.0.weight'] = torch.randn(d_hidden, d_hidden, device=device, dtype=torch.bfloat16)
weights['shared_experts.1.weight'] = torch.randn(d_hidden, d_hidden, device=device, dtype=torch.bfloat16)
weights['shared_experts.2.weight'] = torch.randn(d_hidden, d_hidden, device=device, dtype=torch.bfloat16)
# --- Individual expert weights
for i in range(num_experts):
    weights[f'experts.{i}.0.weight'] = torch.randn(d_hidden, d_hidden, device=device, dtype=torch.bfloat16)
    weights[f'experts.{i}.1.weight'] = torch.randn(d_hidden, d_hidden, device=device, dtype=torch.bfloat16)
    weights[f'experts.{i}.2.weight'] = torch.randn(d_hidden, d_hidden, device=device, dtype=torch.bfloat16)

# 4. Run the MoE layer
output, aux_data = custom_kernel((input_tensor, weights, config))

print("Output shape:", output.shape)
print("Load balancing loss:", aux_data['load_balancing_loss'])
```

### Usage Example: FP8 GEMM

The `amd-fp8-gemm.py` file uses the Triton autotuner to find the best configuration for your hardware.
You can then use the `custom_kernel` wrapper to execute the kernel with the proper arguments.

```python
import torch
from amd_fp8_gemm import custom_kernel

# 1. Define matrix shapes
M, N, K = 4096, 4096, 2048
device = 'cuda'

# 2. Create dummy FP8 tensors and FP32 scales
# (Note: torch doesn't have a native FP8 type, so we use int8 for storage)
a = torch.randint(-128, 127, (M, K), device=device, dtype=torch.int8)
b = torch.randint(-128, 127, (K, N), device=device, dtype=torch.int8)

# Per-block scaling example
a_scale = torch.randn((M, K // 128), device=device, dtype=torch.float32)
b_scale = torch.randn((N // 128, K // 128), device=device, dtype=torch.float32)

c = torch.empty((M, N), device=device, dtype=torch.bfloat16)

# 3. Run the kernel
# The first run will be slow as the autotuner finds the best config
output_c = custom_kernel((a, b, a_scale, b_scale, c))

print("Output shape:", output_c.shape)
print("GEMM executed successfully.")
```

## Acknowledgments

This work was created for the **AMD Inference Sprint**. Special thanks to the organizers and the open-source community for providing the tools and platforms that make such projects possible.
