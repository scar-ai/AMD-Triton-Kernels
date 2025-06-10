#!POPCORN leaderboard amd-mixture-of-experts

# This is a submission template for popcorn leaderboard 'amd-mixture-of-experts'.
# Your task is as follows:
# > For a more complete description, see: https://tinyurl.com/amd-comp-moe
# > Implement a DeepSeek-style Mixture of Experts (MoE) layer for efficient transformer models
# > on a single MI300X device.
# > 
# > MoE is a technique that allows scaling model capacity without proportionally increasing computational costs
# > by using a routing mechanism to selectively activate only a subset of parameters for each token.
# > 
# > Your task:
# > - Implement token routing using a simple softmax-based learned router
# > - Route tokens to the top-k experts based on router probabilities
# > - Process tokens through their assigned experts
# > - Combine expert outputs weighted by router probabilities
# > - Calculate appropriate auxiliary losses for training stability
# > 
# > Input:
# > - `data`: Tuple of (input: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
# >   - input: Input tensor of shape [bs, seq_len, d_hidden]
# >   - weights: Dictionary containing model weights
# >   - config: Dictionary containing model configuration parameters
# > 
# > Output:
# > - Tuple containing:
# >   - output: Processed tensor [bs, seq_len, d_model]
# >   - aux_data: Dictionary with auxiliary data like router probabilities and losses
# The deadline for this leaderboard is 2025-06-02 00:00:00+00:00

# You can automatically route this file to specific GPUs by adding a line
# `#!POPCORN gpus <GPUs>` to the header of this file.
# Happy hacking!

import torch
from typing import Dict, Tuple, Optional

import triton
import triton.language as tl
import aiter.ops.aiter_operator as ops


from task import input_t, output_t

@triton.jit
def silu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    output = x * tl.sigmoid(x.to(tl.float32))
    tl.store(output_ptr + offsets, output, mask=mask)

def triton_silu(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    assert x.is_contiguous(), "Input tensor must be contiguous for triton_silu"
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    if n_elements == 0:
        return output

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    silu_kernel[grid](
        x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return output




@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE:tl.constexpr=4
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am_block = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn_block = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k_iter = tl.arange(0, BLOCK_SIZE_K)

    a_block_ptrs_base = A_ptr + offs_am_block[:, None] * stride_am
    b_block_ptrs_base = B_ptr + offs_bn_block[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start_offset in range(0, K, BLOCK_SIZE_K):
        k_curr_offs = k_start_offset + offs_k_iter
        
        a_ptrs = a_block_ptrs_base + k_curr_offs[None, :] * stride_ak
        b_ptrs = b_block_ptrs_base + k_curr_offs[:, None] * stride_bk
        
        a_mask = (offs_am_block[:, None] < M) & (k_curr_offs[None, :] < K)
        b_mask = (k_curr_offs[:, None] < K) & (offs_bn_block[None, :] < N)
        
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        accumulator += tl.dot(a_tile, b_tile)

    c_val = accumulator.to(C_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c_val, mask=c_mask)

def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K_a = a.shape
    K_b, N = b.shape
    assert K_a == K_b, f"Incompatible K dimensions for matmul: A.shape={a.shape}, B.shape={b.shape}"
    K = K_a

    assert a.is_contiguous() and b.is_contiguous(), "Inputs must be contiguous for triton_matmul"
    
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_SIZE_M = 128 
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64

    grid_0 = triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)
    grid = (grid_0,)

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return c


@triton.jit
def transpose_kernel(
    input_ptr, output_ptr,
    M, N,
    stride_in_m, stride_in_n,
    stride_out_n, stride_out_m,
    TILE_DIM_M: tl.constexpr, TILE_DIM_N: tl.constexpr
):
    pid_m_tile = tl.program_id(0)
    pid_n_tile = tl.program_id(1)

    m_start = pid_m_tile * TILE_DIM_M
    n_start = pid_n_tile * TILE_DIM_N

    offs_m_in_tile = tl.arange(0, TILE_DIM_M)
    offs_n_in_tile = tl.arange(0, TILE_DIM_N)

    x_ptrs = (input_ptr +
              (m_start + offs_m_in_tile[:, None]) * stride_in_m +
              (n_start + offs_n_in_tile[None, :]) * stride_in_n)

    y_ptrs = (output_ptr +
              (n_start + offs_n_in_tile[:, None]) * stride_out_n +
              (m_start + offs_m_in_tile[None, :]) * stride_out_m)

    mask_load = ((m_start + offs_m_in_tile[:, None] < M) &
                 (n_start + offs_n_in_tile[None, :] < N))

    mask_store = ((n_start + offs_n_in_tile[:, None] < N) &
                  (m_start + offs_m_in_tile[None, :] < M))

    tile_data = tl.load(x_ptrs, mask=mask_load, other=0.0)
    tl.store(y_ptrs, tl.trans(tile_data), mask=mask_store)

def triton_transpose(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2, "Transpose is defined for 2D tensors"
    M, N = x.shape
    TILE_DIM_M = 32 
    TILE_DIM_N = 32

    assert x.is_contiguous(), "Input must be contiguous for triton_transpose"
    
    output = torch.empty((N, M), device=x.device, dtype=x.dtype)

    grid = (triton.cdiv(M, TILE_DIM_M), triton.cdiv(N, TILE_DIM_N))
    
    transpose_kernel[grid](
        x, output,
        M, N,
        x.stride(0), x.stride(1),
        output.stride(0), output.stride(1),
        TILE_DIM_M=TILE_DIM_M, TILE_DIM_N=TILE_DIM_N
    )
    return output

@triton.jit
def softmax_fwd_kernel_last_dim(
    input_ptr, output_ptr,
    num_rows, num_cols,
    stride_in_row, stride_in_col,
    stride_out_row, stride_out_col,
    BLOCK_COLS: tl.constexpr
):
    row_idx = tl.program_id(axis=0)
    
    row_start_in_ptr = input_ptr + row_idx * stride_in_row
    row_start_out_ptr = output_ptr + row_idx * stride_out_row

    current_max = -float('inf')
    for col_block_start in range(0, num_cols, BLOCK_COLS):
        col_offsets = col_block_start + tl.arange(0, BLOCK_COLS)
        mask = col_offsets < num_cols
        elements_block = tl.load(row_start_in_ptr + col_offsets * stride_in_col, mask=mask, other=-float('inf'))
        block_max = tl.max(elements_block, axis=0)
        current_max = tl.maximum(current_max, block_max)

    current_sum_exp = 0.0
    for col_block_start in range(0, num_cols, BLOCK_COLS):
        col_offsets = col_block_start + tl.arange(0, BLOCK_COLS)
        mask = col_offsets < num_cols
        elements_block = tl.load(row_start_in_ptr + col_offsets * stride_in_col, mask=mask, other=0.0)
        elements_block_shifted = elements_block - current_max
        exp_elements_block = tl.exp(elements_block_shifted.to(tl.float32))
        current_sum_exp += tl.sum(exp_elements_block, axis=0)

    for col_block_start in range(0, num_cols, BLOCK_COLS):
        col_offsets = col_block_start + tl.arange(0, BLOCK_COLS)
        mask = col_offsets < num_cols
        elements_block = tl.load(row_start_in_ptr + col_offsets * stride_in_col, mask=mask, other=0.0)
        
        elements_block_shifted = elements_block - current_max
        exp_elements_block = tl.exp(elements_block_shifted.to(tl.float32))
        softmax_output = exp_elements_block / current_sum_exp
        
        tl.store(row_start_out_ptr + col_offsets * stride_out_col, softmax_output.to(output_ptr.dtype.element_ty), mask=mask)

def triton_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    assert dim == -1 or dim == x.ndim - 1, "Softmax Triton kernel currently only supports the last dimension"
    
    input_shape = x.shape
    num_rows = x.numel() // input_shape[-1]
    num_cols = input_shape[-1]

    assert x.is_contiguous(), "Input must be contiguous for triton_softmax"
    output = torch.empty_like(x)

    BLOCK_COLS = triton.next_power_of_2(min(num_cols, 1024))

    grid = (num_rows,)

    stride_in_col = x.stride(-1)
    stride_in_row = x.stride(-2) if x.ndim > 1 else num_cols
    
    stride_out_col = output.stride(-1)
    stride_out_row = output.stride(-2) if output.ndim > 1 else num_cols

    softmax_fwd_kernel_last_dim[grid](
        x, output,
        num_rows, num_cols,
        stride_in_row, stride_in_col,
        stride_out_row, stride_out_col,
        BLOCK_COLS=BLOCK_COLS
    )
    return output

@triton.jit
def logsumexp_fwd_kernel_last_dim(
    input_ptr, output_ptr,
    num_rows, num_cols,
    stride_in_row, stride_in_col,
    BLOCK_COLS: tl.constexpr
):
    row_idx = tl.program_id(axis=0)
    row_start_in_ptr = input_ptr + row_idx * stride_in_row

    current_max = -float('inf')
    for col_block_start in range(0, num_cols, BLOCK_COLS):
        col_offsets = col_block_start + tl.arange(0, BLOCK_COLS)
        mask = col_offsets < num_cols
        elements_block = tl.load(row_start_in_ptr + col_offsets * stride_in_col, mask=mask, other=-float('inf'))
        block_max = tl.max(elements_block, axis=0)
        current_max = tl.maximum(current_max, block_max)

    current_sum_exp = 0.0
    for col_block_start in range(0, num_cols, BLOCK_COLS):
        col_offsets = col_block_start + tl.arange(0, BLOCK_COLS)
        mask = col_offsets < num_cols
        elements_block = tl.load(row_start_in_ptr + col_offsets * stride_in_col, mask=mask, other=0.0) 
        elements_block_shifted = elements_block - current_max
        exp_elements_block = tl.exp(elements_block_shifted.to(tl.float32))
        current_sum_exp += tl.sum(exp_elements_block, axis=0)
    
    log_sum_exp_val = tl.log(current_sum_exp) + current_max
    tl.store(output_ptr + row_idx, log_sum_exp_val.to(output_ptr.dtype.element_ty))

def triton_logsumexp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    assert dim == -1 or dim == x.ndim - 1, "LogSumExp Triton kernel currently only supports the last dimension"
    
    input_shape = list(x.shape)
    num_rows = x.numel() // input_shape[-1]
    num_cols = input_shape[-1]

    assert x.is_contiguous(), "Input must be contiguous for triton_logsumexp"
    
    output_shape = input_shape[:-1]
    output = torch.empty(output_shape, device=x.device, dtype=torch.float32)

    BLOCK_COLS = triton.next_power_of_2(min(num_cols, 1024))
    grid = (num_rows,)

    stride_in_col = x.stride(-1)
    stride_in_row = x.stride(-2) if x.ndim > 1 else num_cols
    
    logsumexp_fwd_kernel_last_dim[grid](
        x, output,
        num_rows, num_cols,
        stride_in_row, stride_in_col,
        BLOCK_COLS=BLOCK_COLS
    )
    return output.to(x.dtype)

@triton.jit
def sum_reduction_stage1_kernel(
    input_ptr, partial_sums_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    
    block_sum = 0.0
    for i in range(0, tl.cdiv(n_elements, BLOCK_SIZE * tl.num_programs(0)), 1):
        base_offset = pid * BLOCK_SIZE + i * BLOCK_SIZE * tl.num_programs(0)
        offsets = base_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        elements = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        block_sum += tl.sum(elements.to(tl.float32), axis=0)
            
    tl.store(partial_sums_ptr + pid, block_sum)

@triton.jit
def sum_reduction_stage2_kernel(
    partial_sums_ptr, output_scalar_ptr,
    num_partial_sums,
    BLOCK_SIZE: tl.constexpr
):
    total_sum = 0.0
    for i in range(0, tl.cdiv(num_partial_sums, BLOCK_SIZE), 1):
        offsets = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_partial_sums
        partial_s = tl.load(partial_sums_ptr + offsets, mask=mask, other=0.0)
        total_sum += tl.sum(partial_s.to(tl.float32), axis=0)
        
    tl.store(output_scalar_ptr, total_sum)

def triton_sum_all(x: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    if n_elements == 0:
        return torch.tensor(0.0, device=x.device, dtype=torch.float32)

    input_flat = x.view(-1)

    BLOCK_SIZE_STAGE1 = 1024
    num_blocks_stage1 = triton.cdiv(n_elements, BLOCK_SIZE_STAGE1)
    grid_stage1_x = min(num_blocks_stage1, 65535)

    partial_sums = torch.empty(grid_stage1_x, device=x.device, dtype=torch.float32)
    
    sum_reduction_stage1_kernel[(grid_stage1_x,)](
        input_flat, partial_sums,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE_STAGE1
    )

    num_partial_sums = partial_sums.numel()
    if num_partial_sums == 1:
        total_sum_tensor = partial_sums
    else:
        BLOCK_SIZE_STAGE2 = 1024 
        final_sum_tensor = torch.empty(1, device=x.device, dtype=torch.float32)

        sum_reduction_stage2_kernel[(1,)](
            partial_sums, final_sum_tensor,
            num_partial_sums,
            BLOCK_SIZE=BLOCK_SIZE_STAGE2
        )
        total_sum_tensor = final_sum_tensor
        
    return total_sum_tensor[0]

def triton_mean_all(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return torch.tensor(float('nan'), device=x.device, dtype=x.dtype) 
    
    total_sum = triton_sum_all(x)
    mean_val = total_sum / x.numel()
    return mean_val.to(x.dtype)

@triton.jit
def mean_last_dim_kernel(
    input_ptr, output_ptr,
    num_rows, num_cols,
    stride_in_row, stride_in_col,
    BLOCK_COLS: tl.constexpr
):
    row_idx = tl.program_id(axis=0)
    row_start_in_ptr = input_ptr + row_idx * stride_in_row
    
    current_sum = 0.0
    for col_block_start in range(0, num_cols, BLOCK_COLS):
        col_offsets = col_block_start + tl.arange(0, BLOCK_COLS)
        mask = col_offsets < num_cols
        x = tl.load(row_start_in_ptr + col_offsets*stride_in_col, mask=mask, other=0.0)
        current_sum += tl.sum(x.to(tl.float32), axis=0)
    
    mean_val = current_sum / num_cols
    tl.store(output_ptr + row_idx, mean_val.to(output_ptr.dtype.element_ty))

def triton_mean_last_dim(x: torch.Tensor) -> torch.Tensor:
    input_shape = list(x.shape)
    assert x.ndim >=1
    num_rows = x.numel() // input_shape[-1]
    num_cols = input_shape[-1]

    if num_cols == 0:
        output_shape = input_shape[:-1]
        return torch.full(output_shape, float('nan'), device=x.device, dtype=x.dtype)

    assert x.is_contiguous()
    
    output_shape = input_shape[:-1]
    output = torch.empty(output_shape, device=x.device, dtype=torch.float32) 

    BLOCK_COLS = triton.next_power_of_2(min(num_cols, 1024))
    grid = (num_rows,)

    stride_in_col = x.stride(-1)
    stride_in_row = x.stride(-2) if x.ndim > 1 else num_cols

    mean_last_dim_kernel[grid](
        x, output,
        num_rows, num_cols,
        stride_in_row, stride_in_col,
        BLOCK_COLS=BLOCK_COLS
    )
    return output.to(x.dtype)

@triton.jit
def gather_rows_kernel(
    input_ptr, indices_ptr, output_ptr,
    num_output_rows, D_cols,
    stride_input_row, stride_input_col,
    stride_output_row, stride_output_col,
    BLOCK_SIZE_ROW_OUT: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr
):
    pid_row_out_base = tl.program_id(0)
    pid_col_base = tl.program_id(1)

    offs_row_out = pid_row_out_base * BLOCK_SIZE_ROW_OUT + tl.arange(0, BLOCK_SIZE_ROW_OUT)
    offs_col = pid_col_base * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)

    mask_row_out_valid = offs_row_out < num_output_rows
    mask_col_valid = offs_col < D_cols

    indices_to_gather = tl.load(indices_ptr + offs_row_out, mask=mask_row_out_valid, other=0)

    input_row_starts = input_ptr + indices_to_gather[:, None] * stride_input_row
    input_element_ptrs = input_row_starts + offs_col[None, :] * stride_input_col
    
    output_row_starts = output_ptr + offs_row_out[:, None] * stride_output_row
    output_element_ptrs = output_row_starts + offs_col[None, :] * stride_output_col
    
    final_mask = mask_row_out_valid[:, None] & mask_col_valid[None, :]
    
    gathered_data = tl.load(input_element_ptrs, mask=final_mask, other=0.0)
    tl.store(output_element_ptrs, gathered_data, mask=final_mask)

def triton_gather_rows(input_tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    assert input_tensor.ndim == 2 and indices.ndim == 1, "gather_rows expects 2D input and 1D indices"
    assert input_tensor.is_contiguous() and indices.is_contiguous(), "Inputs must be contiguous"

    num_input_rows, D_cols = input_tensor.shape
    num_output_rows = indices.numel()

    output = torch.empty((num_output_rows, D_cols), device=input_tensor.device, dtype=input_tensor.dtype)

    BLOCK_SIZE_ROW_OUT = 64
    BLOCK_SIZE_COL = 64
    
    grid = (triton.cdiv(num_output_rows, BLOCK_SIZE_ROW_OUT), triton.cdiv(D_cols, BLOCK_SIZE_COL))

    gather_rows_kernel[grid](
        input_tensor, indices, output,
        num_output_rows, D_cols,
        input_tensor.stride(0), input_tensor.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_ROW_OUT=BLOCK_SIZE_ROW_OUT, BLOCK_SIZE_COL=BLOCK_SIZE_COL
    )
    return output

@triton.jit
def scatter_add_rows_kernel(
    values_ptr, indices_ptr, output_ptr,
    num_value_rows, D_cols,
    stride_values_row, stride_values_col,
    stride_output_row, stride_output_col,
    BLOCK_SIZE_ROW_VAL: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr
):
    pid_row_val_base = tl.program_id(0)
    pid_col_base = tl.program_id(1)

    offs_row_val = pid_row_val_base * BLOCK_SIZE_ROW_VAL + tl.arange(0, BLOCK_SIZE_ROW_VAL)
    offs_col = pid_col_base * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)

    mask_row_val_valid = offs_row_val < num_value_rows
    mask_col_valid = offs_col < D_cols
    
    values_row_starts = values_ptr + offs_row_val[:, None] * stride_values_row
    current_values_ptrs = values_row_starts + offs_col[None, :] * stride_values_col
    
    load_mask = mask_row_val_valid[:, None] & mask_col_valid[None, :]
    vals_to_add = tl.load(current_values_ptrs, mask=load_mask, other=0.0)

    target_row_indices = tl.load(indices_ptr + offs_row_val, mask=mask_row_val_valid, other=0)

    output_row_starts = output_ptr + target_row_indices[:, None] * stride_output_row
    output_atomic_ptrs = output_row_starts + offs_col[None, :] * stride_output_col
    
    tl.atomic_add(output_atomic_ptrs, vals_to_add, mask=load_mask)

def triton_scatter_add_rows(output_tensor: torch.Tensor, indices: torch.Tensor, values: torch.Tensor):
    assert output_tensor.ndim == 2 and values.ndim == 2 and indices.ndim == 1
    assert output_tensor.shape[1] == values.shape[1], "Number of columns must match for output and values"
    assert indices.numel() == values.shape[0], "Number of indices must match number of rows in values"
    assert (output_tensor.is_contiguous() and indices.is_contiguous() and values.is_contiguous()), "All tensors must be contiguous for triton_scatter_add_rows"

    num_value_rows, D_cols = values.shape

    BLOCK_SIZE_ROW_VAL = 64
    BLOCK_SIZE_COL = 64
    
    grid = (triton.cdiv(num_value_rows, BLOCK_SIZE_ROW_VAL), triton.cdiv(D_cols, BLOCK_SIZE_COL))

    scatter_add_rows_kernel[grid](
        values, indices, output_tensor,
        num_value_rows, D_cols,
        values.stride(0), values.stride(1),
        output_tensor.stride(0), output_tensor.stride(1),
        BLOCK_SIZE_ROW_VAL=BLOCK_SIZE_ROW_VAL, BLOCK_SIZE_COL=BLOCK_SIZE_COL
    )

@triton.jit
def zeros_kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    zeros_val = tl.zeros((BLOCK_SIZE,), dtype=output_ptr.dtype.element_ty)
    tl.store(output_ptr + offsets, zeros_val, mask=mask)

def triton_zeros_like(x_prototype: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x_prototype)

    n_elements = output.numel()
    BLOCK_SIZE = 1024
    if n_elements == 0: return output
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    zeros_kernel[grid](output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output

def triton_topk_stub(probs: torch.Tensor, k: int, dim: int = -1, sorted: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.topk(probs, k=k, dim=dim, sorted=sorted)

def triton_argsort_stub(x: torch.Tensor, dim: int = -1, descending: bool = False) -> torch.Tensor:
    return torch.argsort(x, dim=dim, descending=descending)

def triton_bincount_stub(x: torch.Tensor, weights: Optional[torch.Tensor]=None, minlength: int=0) -> torch.Tensor:
    return torch.bincount(x, weights=weights, minlength=minlength)

def triton_cumsum_stub(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.cumsum(x, dim=dim)

def expert_ffn_with_all_triton(
    x: torch.Tensor, 
    W_gate_t: torch.Tensor,
    W_up_t: torch.Tensor,
    W_down_t: torch.Tensor
) -> torch.Tensor:
    gate_h = triton_matmul(x, W_gate_t)

    up_h = triton_matmul(x, W_up_t)
    
    activated_gate = triton_silu(gate_h)
    
    fused_h = activated_gate * up_h
    out = triton_matmul(fused_h, W_down_t)
    
    return out

def moe_layer_with_all_triton(
    input_tensor: torch.Tensor, 
    weights: Dict[str, torch.Tensor], 
    config: Dict
) -> Tuple[torch.Tensor, Dict]:
    bs, seq_len, d_hidden = input_tensor.shape
    num_routed_experts = config["n_routed_experts"]
    top_k = config["n_experts_per_token"]
    
    input_tensor_c = input_tensor

    shared_expert_W_gate_t = weights['shared_experts.0.weight']
    shared_expert_W_up_t = weights['shared_experts.1.weight']
    shared_expert_W_down_t = weights['shared_experts.2.weight']
    
    shared_output = expert_ffn_with_all_triton(
        input_tensor_c.view(-1, d_hidden),
        shared_expert_W_gate_t, 
        shared_expert_W_up_t, 
        shared_expert_W_down_t
    ).view(bs, seq_len, d_hidden)

    router_W_g = weights['router.weight']
    router_W_g_transposed_for_matmul = triton_transpose(router_W_g)
    router_logits_flat = ops.mul(input_tensor_c.view(-1, d_hidden), router_W_g_transposed_for_matmul)
    router_logits = router_logits_flat.view(bs, seq_len, num_routed_experts)

    router_probs_float32 = triton_softmax(router_logits.to(torch.float32), dim=-1)
    router_probs = router_probs_float32.to(input_tensor_c.dtype)

    topk_scores, topk_indices = triton_topk_stub(router_probs, k=top_k, dim=-1, sorted=False)

    x_flat = input_tensor_c.view(-1, d_hidden)
    num_tokens = x_flat.shape[0]

    expert_indices_flat = topk_indices.view(-1)
    expert_scores_flat = topk_scores.view(-1, 1)

    original_token_idx_expanded = torch.arange(num_tokens, device=input_tensor_c.device, dtype=torch.long)
    original_token_idx_expanded = original_token_idx_expanded.unsqueeze(1).expand(-1, top_k).reshape(-1)
    
    perm = triton_argsort_stub(expert_indices_flat, dim=-1, descending=False)
    
    sorted_expert_indices = expert_indices_flat[perm]
    sorted_token_indices = original_token_idx_expanded[perm]
    sorted_scores = expert_scores_flat[perm]
    
    batched_expert_inputs = triton_gather_rows(x_flat, sorted_token_indices)

    expert_counts = triton_bincount_stub(sorted_expert_indices, minlength=num_routed_experts)

    zero_offset = torch.tensor([0], device=input_tensor_c.device, dtype=torch.long)
    cumsum_counts = triton_cumsum_stub(expert_counts, dim=0)
    expert_offsets = torch.cat([zero_offset, cumsum_counts[:-1]])

    batched_expert_outputs = triton_zeros_like(batched_expert_inputs)

    for i in range(num_routed_experts):
        start_offset = expert_offsets[i].item()
        num_assignments_for_expert = expert_counts[i].item()

        if num_assignments_for_expert == 0:
            continue

        end_offset = start_offset + num_assignments_for_expert
        current_expert_input_batch = batched_expert_inputs.narrow(0, start_offset, num_assignments_for_expert)

        exp_W_gate_t = weights[f'experts.{i}.0.weight']
        exp_W_up_t   = weights[f'experts.{i}.1.weight']
        exp_W_down_t = weights[f'experts.{i}.2.weight']

        expert_output_for_batch = expert_ffn_with_all_triton(
            current_expert_input_batch,
            exp_W_gate_t,
            exp_W_up_t,
            exp_W_down_t
        )
        batched_expert_outputs.narrow(0, start_offset, num_assignments_for_expert).copy_(expert_output_for_batch)
    
    weighted_expert_outputs = batched_expert_outputs*sorted_scores
    
    final_routed_output_flat = triton_zeros_like(x_flat)
    
    triton_scatter_add_rows(final_routed_output_flat, sorted_token_indices, weighted_expert_outputs)
    
    routed_output = final_routed_output_flat.view(bs, seq_len, d_hidden)
    
    output_tensor = routed_output+shared_output

    aux_data = {}
    aux_data['router_probs'] = router_probs
    aux_data['router_logits'] = router_logits

    router_z_loss_coef = config.get("router_z_loss_coef", 0.001)
    router_logits_flat_for_loss = router_logits.view(-1, num_routed_experts)
    
    log_z = triton_logsumexp(router_logits_flat_for_loss.to(torch.float32), dim=-1)
    
    log_z_squared = log_z * log_z
    mean_log_z_squared = triton_mean_all(log_z_squared)
    router_z_loss = router_z_loss_coef * mean_log_z_squared 
    aux_data['router_z_loss'] = router_z_loss.to(input_tensor_c.dtype)

    avg_router_probs_per_expert = torch.mean(router_probs, dim=(0,1))

    if (num_tokens * top_k) > 0:
        fraction_tokens_per_expert = expert_counts.to(torch.float32) / (num_tokens * top_k)
    else:
        fraction_tokens_per_expert = triton_zeros_like(expert_counts).to(torch.float32)

    load_balance_loss_coef = config.get("load_balance_loss_coef", 0.01)
    prod_for_lb = avg_router_probs_per_expert.to(torch.float32) * fraction_tokens_per_expert
    sum_prod_for_lb = triton_sum_all(prod_for_lb)
    
    load_balancing_loss = load_balance_loss_coef * num_routed_experts * sum_prod_for_lb
    aux_data['load_balancing_loss'] = load_balancing_loss.to(input_tensor_c.dtype)

    aux_data['expert_assignment_counts'] = expert_counts
    aux_data['avg_router_probs_per_expert_stat'] = avg_router_probs_per_expert 
    aux_data['fraction_tokens_per_expert_stat'] = fraction_tokens_per_expert

    return output_tensor

def custom_kernel(data: input_t) -> output_t:
    """
    Submission template for DeepSeek-style Mixture of Experts using PyTorch.
    
    Args:
        data: Tuple of (input: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
            - input: Input tensor of shape [batch_size, seq_len, hidden_size]
            - weights: Dictionary containing model weights
            - config: Dictionary containing model configuration parameters
            
    Returns:
        Tuple containing:
            - output: Processed tensor [batch_size, seq_len, d_model]
            - aux_data: Dictionary with auxiliary data
    """
    input_tensor, weights, config = data

    return moe_layer_with_all_triton(input_tensor, weights, config)