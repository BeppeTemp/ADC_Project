#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <chrono>

#define MASK_SIZE 4
#define MASK_CENTER 1

#define BLOCK_DIM 32
#define TILE_WIDTH 32


void conv_cpu(float* mat_start, float* mask, float* mat_res, int mat_size, int mask_size, int mask_center) {
    #pragma omp parallel for
    for (int mat_row = 0; mat_row < mat_size; mat_row++)
    #pragma omp parallel for
        for (int mat_col = 0; mat_col < mat_size; mat_col++)
            for (int k_row = 0; k_row < mask_size; k_row++)
                for (int k_col = 0; k_col < mask_size; k_col++) {
                    int rel_row = mat_row + (k_row - mask_center);
                    int rel_col = mat_col + (k_col - mask_center);

                    if (rel_row >= 0 && rel_row < mat_size && rel_col >= 0 && rel_col < mat_size) {
                        mat_res[(mat_row * mat_size) + mat_col] += mat_start[(rel_row * mat_size) + rel_col] * mask[(k_row * mask_size) + k_col];
                    }
                }
}

__global__ void ConvolutionKernel(float* mat_start, float* mat_res, const float* Mask, int size) {
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
    if ((row_i >= 0) && (row_i < size) && (col_i >= 0) && (col_i < size)) {
        n_ds[ty][tx] = mat_start[(row_i * size) + col_i];
    }

     __syncthreads();

    // Convolution calculation
    float output = 0.0f;
    if (ty < TILE_WIDTH && tx < TILE_WIDTH) {
        for (int i = 0; i < MASK_SIZE; i++) {
            for (int j = 0; j < MASK_SIZE; j++) {
                output += Mask[(i * MASK_SIZE) + j] * n_ds[i + ty][j + tx];
            }
        }
        if (row_o < size && col_o < size) {
            mat_res[row_o * size + col_o] = output;
        }
    }
}