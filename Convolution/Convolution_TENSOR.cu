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

// MMA matrix tile dimensions.
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

//__global__ void WMMAINT8()
using namespace nvcuda;

void time_stats(float micro_seconds) {
    printf("Execution times:\n");
    printf("    * %.0f μs \n", micro_seconds);
    printf("    * %.2f ms \n", micro_seconds / 1000);
    printf("    * %.2f s \n", micro_seconds / 1000 / 1000);
    printf("\n");
}

__global__ void WMMAF16TensorCore(half* A, half* B, float* C, int size) {
    int ix = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int iy = (blockIdx.y * blockDim.y + threadIdx.y);

    // Both matrix col_major
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // AB = A*B
    int a_col, a_row, b_col, b_row, c_col, c_row;

    a_row = ix * WMMA_M;
    b_col = iy * WMMA_N;  // b_col=iy*N
    
    for (int k = 0; k < size; k += WMMA_K) {
        a_col = b_row = k;  // b_row=k

        if (a_row < size && a_col < size && b_row < size && b_col < size) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, A + a_col + a_row * size, size);
            wmma::load_matrix_sync(b_frag, B + b_col + b_col * size, size);

            // Perform the matrix multiplication
            // ab_frag now holds the result for this warp’s output tile based on the multiplication of A and B.
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // D = AB + C
    c_col = b_col;
    c_row = a_row;
    if (c_row < size && c_col < size) {
        wmma::load_matrix_sync(c_frag, C + c_col + c_row * size, size, wmma::mem_col_major);

        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = acc_frag.x[i] + c_frag.x[i];
        }

        // Store the output
        wmma::store_matrix_sync(C + c_col + c_row * size, c_frag, size, wmma::mem_col_major);
    }
}

int main(void) {
    int sizes[5] = {1024, 2048, 4096, 8192, 16384};

    float *mat_start_host, *mat_res_host, *mask_host;
    half *mat_start_dev, *mat_res_dev, *mask_dev;

    // Mask init and upload
    mask_host = (float*)malloc(MASK_SIZE * MASK_SIZE * sizeof(float));
    for (int i = 0; i < MASK_SIZE * MASK_SIZE; i++) {
        mask_host[i] = 1;
    }
    cudaMalloc((void**)&mask_dev, MASK_SIZE * MASK_SIZE * sizeof(float));
    cudaMemcpy(mask_dev, mask_host, MASK_SIZE * MASK_SIZE * sizeof(float), cudaMemcpyDefault);

    dim3 gridDim, blockDim;

    for (int k = 0; k < 5; k++) {
        mat_start_host = (float*)malloc(sizes[k] * sizes[k] * sizeof(float));
        mat_res_host = (float*)calloc(sizes[k] * sizes[k], sizeof(float));

        for (int i = 0; i < sizes[k] * sizes[k]; i++) {
            mat_start_host[i] = 1;
        }

        cudaMalloc((void**)&mat_start_dev, sizes[k] * sizes[k] * sizeof(float));
        cudaMalloc((void**)&mat_res_dev, sizes[k] * sizes[k] * sizeof(float));

        cudaMemcpy(mat_start_dev, mat_start_host, sizes[k] * sizes[k] * sizeof(float), cudaMemcpyDefault);
        cudaMemcpy(mat_res_dev, mat_res_host, sizes[k] * sizes[k] * sizeof(float), cudaMemcpyDefault);
        cudaMemset(mat_res_dev, 0, sizes[k] * sizes[k] * sizeof(float));

        blockDim.x = BLOCK_DIM;
        blockDim.y = BLOCK_DIM;
        gridDim.x = sizes[k] / blockDim.x + ((sizes[k] % blockDim.x) == 0 ? 0 : 1);
        gridDim.y = sizes[k] / blockDim.y + ((sizes[k] % blockDim.y) == 0 ? 0 : 1);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        WMMAF16TensorCore<<<gridDim, blockDim>>>();
        cudaEventRecord(stop);

        cudaMemcpy(mat_res_host, mat_res_dev, sizes[k] * sizes[k] * sizeof(float), cudaMemcpyDeviceToHost);

        // printMat(mat_res_host, sizes[k]);

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

    return 0;
}