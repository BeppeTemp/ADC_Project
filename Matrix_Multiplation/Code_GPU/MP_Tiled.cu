#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <chrono>

#define BLOCK_DIM 16

#define TILE_WIDTH 16

using namespace std::chrono;

void time_stats(float micro_seconds) {
    printf("Execution times:\n");
    printf("    * %.0f μs \n", micro_seconds);
    printf("    * %.2f ms \n", micro_seconds / 1000);
    printf("    * %.2f s \n", micro_seconds / 1000 / 1000);
    printf("\n");
}


__global__ void MatrixMulKernelTiled(float *mat_a, float *mat_b, float *res_mat, int size) {
    __shared__ float M_ds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float N_ds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    if ((Row < size) && (Col < size)) {
        float Pvalue = 0;
        for (int m = 0; m <= size / TILE_WIDTH; ++m) {
            M_ds[ty][tx] = mat_a[Row * size + m * TILE_WIDTH + tx];
            N_ds[ty][tx] = mat_b[(m * TILE_WIDTH + ty) * size + Col];

            __syncthreads();

            for (int k = 0; k < TILE_WIDTH; ++k) {
                Pvalue += M_ds[ty][k] * N_ds[k][tx];
            }
            __syncthreads();
        }

        res_mat[Row * size + Col] = Pvalue;
    }
}

int main(void) {
    int sizes[5] = {1024, 2048, 4096, 8192, 16384};

    float *mat_a_host, *mat_b_host, *mat_res_host_gpu;
    float *mat_a_dev, *mat_b_dev, *mat_res_dev;
    dim3 gridDim, blockDim;

    for (int i = 0; i < 5; i++) {
        long nBytes = sizes[i] * sizes[i] * sizeof(float);

        mat_a_host = (float*)malloc(nBytes);
        mat_b_host = (float*)malloc(nBytes);
        mat_res_host_gpu = (float*)malloc(nBytes);

        cudaMalloc((void**)&mat_a_dev, nBytes);
        cudaMalloc((void**)&mat_b_dev, nBytes);
        cudaMalloc((void**)&mat_res_dev, nBytes);

        for (int j = 0; j < sizes[i] * sizes[i]; j++) {
            mat_a_host[j] = 1;
            mat_b_host[j] = 1;
        }

        cudaMemcpy(mat_a_dev, mat_a_host, nBytes, cudaMemcpyDefault);
        cudaMemcpy(mat_b_dev, mat_b_host, nBytes, cudaMemcpyDefault);
        cudaMemset(mat_res_dev, 0, nBytes);

        blockDim.x = BLOCK_DIM;
        blockDim.y = BLOCK_DIM;
        gridDim.x = sizes[i] / blockDim.x + ((sizes[i] % blockDim.x) == 0 ? 0 : 1);
        gridDim.y = sizes[i] / blockDim.y + ((sizes[i] % blockDim.y) == 0 ? 0 : 1);

        auto start = high_resolution_clock::now();
        MatrixMulKernelTiled<<<gridDim, blockDim>>>(mat_a_dev, mat_b_dev, mat_res_dev, sizes[i]);
        cudaDeviceSynchronize();
        auto stop = high_resolution_clock::now();

        cudaMemcpy(mat_res_host_gpu, mat_res_dev, nBytes, cudaMemcpyDeviceToHost);

        long check = 0;
        for (int k = 0; k < sizes[i] * sizes[i]; k++) {
            check += (long) mat_res_host_gpu[i];
        }

        printf("Matrix size: %d x %d \n", sizes[i], sizes[i]);
        printf("Block size: %d x %d = %d\n", BLOCK_DIM, BLOCK_DIM, BLOCK_DIM * BLOCK_DIM);
        printf("Check: %ld\n", check);
        time_stats(duration_cast<microseconds>(stop - start).count());

        free(mat_a_host);
        free(mat_b_host);

        cudaFree(mat_a_dev);
        cudaFree(mat_b_dev);
        cudaFree(mat_res_dev);
    }

    return 0;
}