#include <cuda.h>
#include <mma.h>
#include <omp.h>
#include <iostream>

#include "../include/matrix_op.cuh"

using namespace nvcuda;

// Kernels
__global__ void mm_tiled_kernel(float* mat_a, float* mat_b, float* res_mat, int size) {
    __shared__ float m_a_sh[BLOCK_DIM][BLOCK_DIM];
    __shared__ float m_b_sh[BLOCK_DIM][BLOCK_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int mat_a_begin = size * BLOCK_DIM * by;
    int mat_a_end = mat_a_begin + size - 1;
    int mat_b_begin = BLOCK_DIM * bx;

    int mat_b_step = BLOCK_DIM * size;

    float temp = 0;
    for (int a = mat_a_begin, b = mat_b_begin; a <= mat_a_end; a += BLOCK_DIM, b += mat_b_step) {
        m_a_sh[ty][tx] = mat_a[a + size * ty + tx];
        m_b_sh[ty][tx] = mat_b[b + size * ty + tx];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_DIM; ++k) {
            temp += m_a_sh[ty][k] * m_b_sh[k][tx];
        }

        __syncthreads();
    }

    int c = size * BLOCK_DIM * by + BLOCK_DIM * bx;
    res_mat[c + size * ty + tx] = temp;
}
__global__ void mm_tensor_kernel(half* mat_a, half* mat_b, float* res_mat, int size) {
    // Tile using a 2D grid
    int tile_row = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int tile_col = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // Parte da 0 e sale di 16 alla volta
    for (int i = 0; i < size; i += WMMA_K) {
        int aCol = i;  // 0
        int aRow = tile_row * WMMA_M;

        int bCol = tile_col * WMMA_N;
        int bRow = i;  // 0

        // Bounds checking
        if (aRow < size && aCol < size && bRow < size && bCol < size) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, mat_a + (aRow * size) + aCol, size);  // mat_a[aRow, aCol]
            wmma::load_matrix_sync(b_frag, mat_b + (bRow * size) + bCol, size);  // mat_b[bRow, bCol]

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    int cCol = tile_row * WMMA_N;
    int cRow = tile_col * WMMA_M;

    // Store the output
    wmma::store_matrix_sync(res_mat + (cRow * size) + cCol, acc_frag, size, wmma::mem_row_major);  // mat_c[cRow, cCol]
}

// Functions
double mm_cpu(float* mat_a, float* mat_b, float* mat_res, int size) {
    double t_init = omp_get_wtime();

#pragma omp parallel for collapse(3)
    for (int mat_row = 0; mat_row < size; mat_row++)
        for (int mat_col = 0; mat_col < size; mat_col++)
            for (int k_row = 0; k_row < size; k_row++)
                mat_res[mat_row * size + mat_col] += mat_a[mat_row * size + k_row] * mat_b[k_row * size + mat_col];

    return omp_get_wtime() - t_init;
}
double mm_gpu(float* mat_a, float* mat_b, float* mat_res, int size) {
    float *res_mat_dev, *mat_a_dev, *mat_b_dev;

    cudaMalloc((void**)&mat_a_dev, size * size * sizeof(float));
    cudaMalloc((void**)&mat_b_dev, size * size * sizeof(float));
    cudaMalloc((void**)&res_mat_dev, size * size * sizeof(float));

    cudaMemcpy(mat_a_dev, mat_a, size * size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mat_b_dev, mat_b, size * size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(res_mat_dev, 0, size * size * sizeof(float));

    dim3 gridDim, blockDim;

    blockDim.x = BLOCK_DIM;
    blockDim.y = BLOCK_DIM;
    gridDim.x = size / blockDim.x + ((size % blockDim.x) == 0 ? 0 : 1);
    gridDim.y = size / blockDim.y + ((size % blockDim.y) == 0 ? 0 : 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mm_tiled_kernel<<<gridDim, blockDim>>>(mat_a_dev, mat_b_dev, res_mat_dev, size);
    cudaEventRecord(stop);

    cudaMemcpy(mat_res, res_mat_dev, size * size * sizeof(float), cudaMemcpyDeviceToHost);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    cudaFree(mat_a_dev);
    cudaFree(mat_b_dev);
    cudaFree(res_mat_dev);

    return elapsed;
}
double mm_tensor(half* mat_a, half* mat_b, float* mat_res, int size) {
    half *mat_b_dev, *mat_a_dev;
    float* res_mat_dev;

    cudaMalloc((void**)&mat_a_dev, size * size * sizeof(half));
    cudaMalloc((void**)&mat_b_dev, size * size * sizeof(half));
    cudaMalloc((void**)&res_mat_dev, size * size * sizeof(float));

    cudaMemcpy(mat_a_dev, mat_a, size * size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(mat_b_dev, mat_b, size * size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(res_mat_dev, 0, size * size * sizeof(float));

    dim3 gridDim, blockDim;

    blockDim.x = 128;
    blockDim.y = 4;
    gridDim.x = (size + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    gridDim.y = (size + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mm_tensor_kernel<<<gridDim, blockDim>>>(mat_a_dev, mat_b_dev, res_mat_dev, size);
    cudaEventRecord(stop);

    cudaMemcpy(mat_res, res_mat_dev, size * size * sizeof(float), cudaMemcpyDeviceToHost);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    cudaFree(mat_a_dev);
    cudaFree(mat_b_dev);
    cudaFree(res_mat_dev);

    return elapsed;
}