#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define WARP_SIZE 32
#define BLOCK_DIM 16

#define PRINT_GREEN(str) printf("\x1b[32m%s\x1b[0m", str);
#define PRINT_RED(str) printf("\x1b[31m%s\x1b[0m", str);

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

    half *mat_a_host, *mat_b_host;
    float* mat_res_host_gpu;
    half *mat_a_dev, *mat_b_dev;
    float* mat_res_dev;
    dim3 gridDim, blockDim;

    for (int i = 0; i < 5; i++) {
        long nBytes = sizes[i] * sizes[i] * sizeof(float);

        mat_a_host = (half*)malloc(nBytes);
        mat_b_host = (half*)malloc(nBytes);
        mat_res_host_gpu = (float*)malloc(nBytes);

        cudaMalloc((void**)&mat_a_dev, nBytes);
        cudaMalloc((void**)&mat_b_dev, nBytes);
        cudaMalloc((void**)&mat_res_dev, nBytes);

        for (int j = 0; j < sizes[i] * sizes[i]; j++) {
            mat_a_host[j] = __float2half(1);
            mat_b_host[j] = __float2half(1);
        }

        cudaMemcpy(mat_a_dev, mat_a_host, nBytes, cudaMemcpyDefault);
        cudaMemcpy(mat_b_dev, mat_b_host, nBytes, cudaMemcpyDefault);
        cudaMemset(mat_res_dev, 0, nBytes);

        blockDim.x = 128;
        blockDim.y = 4;
        gridDim.x = (sizes[i] + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
        gridDim.y = (sizes[i] + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        WMMAF16TensorCore<<<gridDim, blockDim>>>(mat_a_dev, mat_b_dev, mat_res_dev, sizes[i]);
        cudaEventRecord(stop);

        cudaMemcpy(mat_res_host_gpu, mat_res_dev, nBytes, cudaMemcpyDeviceToHost);

        bool check = true;
        for (int k = 0; k < sizes[i] * sizes[i]; k++) {
            if (mat_res_host_gpu[i] != sizes[i])
                check = false;
        }

        printf("Matrix size: %d x %d \n", sizes[i], sizes[i]);
        printf("Block size: %d x %d = %d\n", BLOCK_DIM, BLOCK_DIM, BLOCK_DIM * BLOCK_DIM);
        printf("Check: ");
        if (check) {
            PRINT_GREEN("Verified\n");
        } else {
            PRINT_RED("Error\n");
        }
        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        time_stats(elapsed);

        free(mat_a_host);
        free(mat_b_host);

        cudaFree(mat_a_dev);
        cudaFree(mat_b_dev);
        cudaFree(mat_res_dev);
    }

    return 0;
}