#!POPCORN leaderboard amd-fp8-mm

from task import input_t, output_t
import triton
import triton.language as tl

@triton.autotune(
    configs=[triton.Config(kwargs={'BLOCK_M': m, 'BLOCK_N':n, 'BLOCK_K':k,'GROUP_SIZE': g}, num_warps=w, num_stages=s)
    for m in [32, 64, 128, 256]
    for n in [32, 64, 128, 256]
    for k in [32, 64, 128]
    for g in [2, 4, 8, 16]
    for w in [2, 4, 8, 16]
    for s in [1, 2, 3]],
    key=['m', 'n', 'k']
)

@triton.jit
def kernel(
    a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr,
    m, n, k,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_as_m, stride_as_k,
    stride_bs_n, stride_bs_k,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_SIZE: tl.constexpr, SCALE_GRANULARITY:tl.constexpr = 128
):

    scale_k_ratio = SCALE_GRANULARITY // BLOCK_K
    num_scale_k = tl.cdiv(k, SCALE_GRANULARITY)
    n_scale_blocks = tl.cdiv(n, SCALE_GRANULARITY)

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(m, BLOCK_M)
    num_pid_n = tl.cdiv(n, BLOCK_N)
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    scale_n_idx = (pid_n * BLOCK_N) // SCALE_GRANULARITY
    scale_n_idx = tl.minimum(scale_n_idx, n_scale_blocks - 1)

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(m, k),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(0, 1)
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(k, n),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0)
    )

    a_scale_block_ptr = tl.make_block_ptr(
        base=a_scale_ptr,
        shape=(m, num_scale_k),
        strides=(stride_as_m, stride_as_k),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, 1),
        order=(1, 0)
    )

    b_scale_block_ptr = tl.make_block_ptr(
        base=b_scale_ptr,
        shape=(n_scale_blocks, num_scale_k),
        strides=(stride_bs_n, stride_bs_k),
        offsets=(scale_n_idx, 0),
        block_shape=(1, 1),
        order=(0, 1)
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for scale_k_idx in range(num_scale_k):
        a_scale = tl.load(tl.advance(a_scale_block_ptr, (0, scale_k_idx)))
        b_scale = tl.load(tl.advance(b_scale_block_ptr, (0, scale_k_idx)))

        for sub_k in range(scale_k_ratio):
            k_offset = scale_k_idx * SCALE_GRANULARITY + sub_k * BLOCK_K
            a = tl.load(tl.advance(a_block_ptr, (0, k_offset)))
            b = tl.load(tl.advance(b_block_ptr, (k_offset, 0)), boundary_check=(1, 0))
            
            acc += tl.dot(a, b) * a_scale * b_scale

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask)

def custom_kernel(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    m, k = a.shape
    n = b.shape[0]
    grid = lambda meta: (triton.cdiv(m, meta['BLOCK_M']) * triton.cdiv(n, meta['BLOCK_N']),)
    kernel[grid](a, b, a_scale, b_scale, c,
                 m, n, k,
                 a.stride(0), a.stride(1),
                 b.stride(0), b.stride(1),
                 a_scale.stride(0), a_scale.stride(1),
                 b_scale.stride(0), b_scale.stride(1),
                 c.stride(0), c.stride(1))
    return c