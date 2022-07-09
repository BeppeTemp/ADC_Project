#include <cuda.h>
#include <omp.h>
#include <iostream>

#include "../include/conv_op.cuh"

// Kernels
__global__ void conv_kernel(float* mat_start, const float* mask, float* mat_res, int mat_size) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Coordinate result
    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;

    // Coordinate start
    int row_i = row_o - MASK_CENTER;
    int col_i = col_o - MASK_CENTER;

    // Tile in shared memory
    __shared__ float n_ds[TILE_WIDTH + MASK_SIZE * MASK_SIZE - 1][TILE_WIDTH + MASK_SIZE * MASK_SIZE - 1];

    // Tile cooperative upload
    if ((row_i >= 0) && (row_i < mat_size) && (col_i >= 0) && (col_i < mat_size)) {
        n_ds[ty][tx] = mat_start[(row_i * mat_size) + col_i];
    }

     __syncthreads();

    // Convolution calculation
    float output = 0.0f;
    if (ty < TILE_WIDTH && tx < TILE_WIDTH) {
        for (int i = 0; i < MASK_SIZE; i++) {
            for (int j = 0; j < MASK_SIZE; j++) {
                output += mask[(i * MASK_SIZE) + j] * n_ds[i + ty][j + tx];
            }
        }
        if (row_o < mat_size && col_o < mat_size) {      
            mat_res[row_o * mat_size + col_o] = output;
        }
    }
}

// Functions
double conv_cpu(float* mat_start, float* mask, float* mat_res, int mat_size) {
    double t_init = omp_get_wtime();

    #pragma omp parallel for
    for (int mat_row = 0; mat_row < mat_size; mat_row++)
    #pragma omp parallel for
        for (int mat_col = 0; mat_col < mat_size; mat_col++)
            for (int k_row = 0; k_row < MASK_SIZE; k_row++)
                for (int k_col = 0; k_col < MASK_SIZE; k_col++) {
                    int rel_row = mat_row + (k_row - MASK_CENTER);
                    int rel_col = mat_col + (k_col - MASK_CENTER);

                    if (rel_row >= 0 && rel_row < mat_size && rel_col >= 0 && rel_col < mat_size) {
                        mat_res[(mat_row * mat_size) + mat_col] += mat_start[(rel_row * mat_size) + rel_col] * mask[(k_row * MASK_SIZE) + k_col];
                    }
                }

    return omp_get_wtime() - t_init;
}
double conv_gpu(float* mat_start, float* mask, float* mat_res, int mat_size) {
    float *mat_start_dev, *mask_dev, *mat_res_dev;

    cudaMalloc((void**)&mat_start_dev, mat_size * mat_size * sizeof(float));
    cudaMalloc((void**)&mask_dev, MASK_SIZE * MASK_SIZE * sizeof(float));
    cudaMalloc((void**)&mat_res_dev, mat_size * mat_size * sizeof(float));

    cudaMemcpy(mat_start_dev, mat_start, mat_size * mat_size * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(mask_dev, mask, MASK_SIZE * MASK_SIZE * sizeof(float), cudaMemcpyDefault);
    cudaMemset(mat_res_dev, 0, mat_size * mat_size * sizeof(float));

    dim3 gridDim, blockDim;

    blockDim.x = BLOCK_DIM;
    blockDim.y = BLOCK_DIM;
    gridDim.x = mat_size / blockDim.x + ((mat_size % blockDim.x) == 0 ? 0 : 1);
    gridDim.y = mat_size / blockDim.y + ((mat_size % blockDim.y) == 0 ? 0 : 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    conv_kernel<<<gridDim, blockDim>>>(mat_start_dev, mask_dev, mat_res_dev, mat_size);
    cudaEventRecord(stop);

    cudaMemcpy(mat_res, mat_res_dev, mat_size * mat_size * sizeof(float), cudaMemcpyDeviceToHost);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    cudaFree(mat_start_dev);
    cudaFree(mask_dev);
    cudaFree(mat_res_dev);

    return elapsed;
}