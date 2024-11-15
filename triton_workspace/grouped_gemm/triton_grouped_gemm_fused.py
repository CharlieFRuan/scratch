"""
Start with the follwing implementation, but fuse the inner 2 loops for better pipelining.
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
    # It is NUM_SM behind the actual tile id, so we can add it in prologue in the for loop.
    # Adding it in the epilogue disables pipelining for some reason.
    tile_idx = tl.program_id(0) - NUM_SM
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

        # number of tiles I will compute for problem g
        # Equivalent to `len(range(tile_idx, last_problem_end + num_tiles, NUM_SM))`
        end_tile_idx = last_problem_end + num_tiles
        start_tile_idx = tile_idx + NUM_SM
        if end_tile_idx > start_tile_idx:
            num_tiles_for_me = (end_tile_idx - start_tile_idx - 1) // NUM_SM + 1
        else:
            num_tiles_for_me = 0

        # Total number of k tiles to reduce
        k_tiles = tl.cdiv(k, BLOCK_SIZE_K)
        # Current k tile index
        ki = -1

        # Dummy initializations that will be overriden in the loop
        offs_am = tl.arange(0, BLOCK_SIZE_M)
        offs_bn = tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        tile_m_idx = 0
        tile_n_idx = 0

        # Each iteration reduces a K tile for output tile tile_idx with regular gemm
        for _ in range(0, k_tiles * num_tiles_for_me):
            ki = tl.where(ki == k_tiles - 1, 0, ki + 1)  # advance ki by 1, or reset to 0
            if ki == 0:
                tile_idx += NUM_SM  # go to the next tile by advancing NUM_SM
                # figure out tile coordinates
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx = tile_idx_in_gemm // num_n_tiles
                tile_n_idx = tile_idx_in_gemm % num_n_tiles

                # Get starting pointer for this new output tile
                offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
                offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]

            # hint to Triton compiler to do proper loop pipelining
            # NOTE(Charlie): without it, pipelining is not triggered.
            tl.multiple_of(a_ptrs, [16, 16])
            tl.multiple_of(b_ptrs, [16, 16])

            # assume full tile for now
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            accumulator = tl.dot(a, b, accumulator)

            if ki == k_tiles - 1:
                offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]
                c = accumulator.to(tl.float16)
                tl.store(c_ptrs, c)  # assumes full tile for now
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


if __name__ == "__main__":
    test_grouped_gemm(grouped_matmul_kernel)
