"""
Follows the implementation in
https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html
"""

import torch
import triton
import triton.language as tl

from test_utils import test_grouped_gemm

@triton.jit
def grouped_matmul_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # the tile currently I am about to work on; global tile id, across all group_size problems
    tile_idx = tl.program_id(0)
    # the last problem g's last tile_idx
    last_problem_end = 0

    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles  # total number of tiles in problem g

        # Load pointrs for the problem g
        k = gk
        lda = tl.load(g_lds + g * 3)
        ldb = tl.load(g_lds + g * 3 + 1)
        ldc = tl.load(g_lds + g * 3 + 2)
        a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
        b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
        c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))

        # iterate through the tiles in the current gemm problem
        # What does `tile_idx >= last_problem_end` check?
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                # assume full tile for now
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb
            c = accumulator.to(tl.float16)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]
            tl.store(c_ptrs, c)  # assumes full tile for now
            tile_idx += NUM_SM  # go to the next tile by advancing NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


if __name__ == "__main__":
    test_grouped_gemm(grouped_matmul_kernel)
