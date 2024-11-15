import copy

import torch

import triton

from triton_grouped_gemm import grouped_matmul_kernel as grouped_matmul_kernel_triton
from triton_grouped_gemm_fused import grouped_matmul_kernel as grouped_matmul_kernel_fused_triton

# Configurations
base_config = {
    "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
}
triton_persistent_config = {**base_config}

NUM_SM = torch.cuda.get_device_properties("cuda").multi_processor_count


# -------------------
# Triton Grouped GEMM
# -------------------

def grouped_gemm_triton_persistent(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size, group_C):
    # Launch kernel
    grid = lambda META: (NUM_SM, )
    grouped_matmul_kernel_triton[grid](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        group_size,
        NUM_SM=NUM_SM,
        **triton_persistent_config,
    )

    return group_C

def grouped_gemm_triton_persistent_fused(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size, group_C):
    # Launch kernel
    grid = lambda META: (NUM_SM, )
    grouped_matmul_kernel_fused_triton[grid](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        group_size,
        NUM_SM=NUM_SM,
        **triton_persistent_config,
    )

    return group_C


# ------------
# Benchmarking
# ------------

def get_group_AB_from_MNKs(group_m, group_n, group_k):
    group_A = []
    group_B = []
    assert len(group_m) == len(group_n)
    assert len(group_n) == len(group_k)
    group_size = len(group_m)
    for i in range(group_size):
        M = group_m[i]
        N = group_n[i]
        K = group_k[i]
        A = torch.rand((M, K), device="cuda", dtype=torch.float16)
        B = torch.rand((K, N), device="cuda", dtype=torch.float16)
        group_A.append(A)
        group_B.append(B)
    return group_A, group_B


def get_kernel_params_helper(group_A, group_B):
    device = torch.device('cuda')
    assert len(group_A) == len(group_B)
    group_size = len(group_A)

    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C = []
    for i in range(group_size):
        A = group_A[i]
        B = group_B[i]
        assert A.shape[1] == B.shape[0]
        M, K = A.shape
        K, N = B.shape
        C = torch.empty((M, N), device=device, dtype=A.dtype)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]

    # note these are device tensors
    d_a_ptrs = torch.tensor(A_addrs, device=device)
    d_b_ptrs = torch.tensor(B_addrs, device=device)
    d_c_ptrs = torch.tensor(C_addrs, device=device)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=device)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=device)

    return d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size, group_C


if __name__ == "__main__":
    grouped_gemm_fns = [
        grouped_gemm_triton_persistent,
        grouped_gemm_triton_persistent_fused,
    ]

    torch.manual_seed(0)

    # mnk_sizes = [
    #     (4096, 4096, 4096),
    #     (4096, 4096, 2048),
    #     (4096, 4096, 1024),
    #     (4096, 4096, 512),
    #     (4096, 4096, 256),
    # ]
    # for mnk_size in mnk_sizes:
    #     print(f"MNK: {mnk_size}")
    #     group_m = [mnk_size[0]]
    #     group_n = [mnk_size[1]]
    #     group_k = [mnk_size[2]]

    # Test 5 GEMMs of M=N=K, 4 GEMMs having 2K, 1 GEMM having a different size
    diff_GEMM_sizes = [512, 1024, 2048, 4096, 8192]
    for diff_GEMM_size in diff_GEMM_sizes:
        print(f"4 2048-GEMM, 1 {diff_GEMM_size}-GEMM")
        group_m = [2048, 2048, 2048, 2048, diff_GEMM_size]
        group_n = copy.deepcopy(group_m)
        group_k = copy.deepcopy(group_m)
        for grouped_gemm_fn in grouped_gemm_fns:
            # Get input parameters
            group_A, group_B = get_group_AB_from_MNKs(group_m, group_n, group_k)
            d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size, group_C = get_kernel_params_helper(group_A, group_B)

            # Benchmark the kernel
            ms = triton.testing.do_bench(
                lambda: grouped_gemm_fn(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size, group_C),
                warmup=25,
                rep=100,
            )

            time_us = ms * 1000
            print(f"{grouped_gemm_fn.__name__}: {time_us:.2f} us")
        print()
