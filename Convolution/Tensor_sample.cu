#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MASK_SIZE 4
#define MASK_CENTER 1

#define WARP_SIZE 32
#define BLOCK_DIM 16

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

using namespace nvcuda;

void time_stats(float micro_seconds) {
    printf("Execution times:\n");
    printf("    * %.0f μs \n", micro_seconds);
    printf("    * %.2f ms \n", micro_seconds / 1000);
    printf("    * %.2f s \n", micro_seconds / 1000 / 1000);
    printf("\n");
}

void printMat(float* mat, int size) {
    // Print the entire matrix
    printf("\n");
    for (int i = 0; i < (size * size); i++) {
        printf("|");
        printf("%05.2f", mat[i]);
        if (((i + 1) % (size) == 0) && (i != 0))
            printf("|\n");
        if ((size * size) == 1)
            printf("|\n");
        if (size == 1 && ((i == 0)))
            printf("|\n");
    }
    printf("\n");
}

__global__ void ConvolutionKernelTensor(half* mat_a, half* mat_b, float* mat_c, int size) {
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Load the inputs
    wmma::load_matrix_sync(a_frag, mat_a, 16);
    wmma::load_matrix_sync(b_frag, mat_b, 16);

    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the output
    wmma::store_matrix_sync(mat_c, c_frag, 16, wmma::mem_row_major);
}

int main(void) {
    int sizes[1] = {4};

    half *mat_start_host, *mask_host;
    half *mat_start_dev, *mask_dev;
    float* mat_res_host;
    float* mat_res_dev;

    // Mask init and upload
    mask_host = (half*)malloc(MASK_SIZE * MASK_SIZE * sizeof(half));
    for (int i = 0; i < MASK_SIZE * MASK_SIZE; i++) {
        mask_host[i] = __float2half(1);
    }
    cudaMalloc((void**)&mask_dev, MASK_SIZE * MASK_SIZE * sizeof(half));
    cudaMemcpy(mask_dev, mask_host, MASK_SIZE * MASK_SIZE * sizeof(half), cudaMemcpyDefault);

    dim3 gridDim, blockDim;

    for (int k = 0; k < 1; k++) {
        mat_start_host = (half*)malloc(sizes[k] * sizes[k] * sizeof(half));
        mat_res_host = (float*)calloc(sizes[k] * sizes[k], sizeof(float));

        for (int i = 0; i < sizes[k] * sizes[k]; i++) {
            mat_start_host[i] = __float2half(1);
        }

        cudaMalloc((void**)&mat_start_dev, sizes[k] * sizes[k] * sizeof(half));
        cudaMalloc((void**)&mat_res_dev, sizes[k] * sizes[k] * sizeof(float));

        cudaMemcpy(mat_start_dev, mat_start_host, sizes[k] * sizes[k] * sizeof(half), cudaMemcpyDefault);
        cudaMemcpy(mat_res_dev, mat_res_host, sizes[k] * sizes[k] * sizeof(half), cudaMemcpyDefault);
        cudaMemset(mat_res_dev, 0, sizes[k] * sizes[k] * sizeof(half));

        blockDim.x = BLOCK_DIM;
        blockDim.y = BLOCK_DIM;
        gridDim.x = sizes[k] / blockDim.x + ((sizes[k] % blockDim.x) == 0 ? 0 : 1);
        gridDim.y = sizes[k] / blockDim.y + ((sizes[k] % blockDim.y) == 0 ? 0 : 1);

        printf("Griglia: %d, %d\n", gridDim.x, gridDim.y);
        printf("Blocco: %d, %d\n", blockDim.x, blockDim.y);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        ConvolutionKernelTensor<<<gridDim, blockDim>>>(mat_start_dev, mask_dev, mat_res_dev, sizes[k]);
        cudaEventRecord(stop);

        cudaMemcpy(mat_res_host, mat_res_dev, sizes[k] * sizes[k] * sizeof(float), cudaMemcpyDeviceToHost);

        printMat(mat_res_host, sizes[k]);

        printf("Matrix size: %d x %d \n", sizes[k], sizes[k]);
        printf("Block size: %d x %d = %d\n", BLOCK_DIM, BLOCK_DIM, BLOCK_DIM * BLOCK_DIM);
        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        time_stats(elapsed);

        free(mat_start_host);
        free(mat_res_host);

        cudaFree(mat_start_dev);
        cudaFree(mat_res_dev);
    }
    free(mask_host);
    cudaFree(mask_dev);
}