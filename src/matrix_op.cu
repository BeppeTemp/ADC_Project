#include <cuda.h>
#include <mma.h>
#include <omp.h>
#include <iostream>

#include "../include/matrix_op.cuh"

using namespace nvcuda;

__global__ void mm_tiled_kernel(float* mat_a, float* mat_b, float* res_mat, int size) {
    __shared__ float sA[BLOCK_DIM][BLOCK_DIM];
    __shared__ float sB[BLOCK_DIM][BLOCK_DIM];

    int Row = blockDim.y * blockIdx.y + threadIdx.y;
    int Col = blockDim.x * blockIdx.x + threadIdx.x;

    float Cvalue = 0.0;
    sA[threadIdx.y][threadIdx.x] = 0.0;
    sB[threadIdx.y][threadIdx.x] = 0.0;

    for (int k = 0; k < (((size - 1) / BLOCK_DIM) + 1); k++) {
        if ((Row < size) && (threadIdx.x + (k * BLOCK_DIM)) < size) {
            sA[threadIdx.y][threadIdx.x] = mat_a[(Row * size) + threadIdx.x + (k * BLOCK_DIM)];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0;
        }
        if (Col < size && (threadIdx.y + k * BLOCK_DIM) < size) {
            sB[threadIdx.y][threadIdx.x] = mat_b[(threadIdx.y + k * BLOCK_DIM) * size + Col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < BLOCK_DIM; ++j) {
            Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
    }
    if (Row < size && Col < size) {
        res_mat[Row * size + Col] = Cvalue;
    }
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

    /*#pragma omp parallel for
    for (int ih = 0; ih < size; ih += TILE_CPU)
        #pragma omp parallel for
        for (int jh = 0; jh < size; jh += TILE_CPU)
            for (int kh = 0; kh < size; kh += TILE_CPU)
                for (int il = 0; il < TILE_CPU; il++)
                    for (int kl = 0; kl < TILE_CPU; kl++)
                        for (int jl = 0; jl < TILE_CPU; jl++)
                            mat_res[(ih + il) * TILE_CPU + (jh + jl)] += mat_a[(ih + il) * TILE_CPU + (kh + kl)] * mat_b[(kh + kl) * TILE_CPU + (jh + jl)];
*/
    for (int m = 0; m < size; m++) {
        for (int k = 0; k < size; k++) {
            for (int n = 0; n < size; n++) {
                mat_res[m * size + n] += mat_a[m * size + k] * mat_b[k *size + n];
            }
        }
    }
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

// Matrix Checker, abbiamo sostituito mat_res[i]!=16 con !=size
bool mm_checker(float* mat_res, int size) {
    for (int i = 0; i < size * size; i++)
        if (mat_res[i] != size)
            return false;
    return true;
}