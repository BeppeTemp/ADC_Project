#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <chrono>

#define PRINT_GREEN(str) printf("\x1b[32m%s\x1b[0m", str);
#define PRINT_RED(str) printf("\x1b[31m%s\x1b[0m", str);

// TODO compute profiler
#define BLOCK_DIM 32

using namespace std::chrono;

void time_stats(float micro_seconds) {
    printf("Execution times:\n");
    printf("    * %.0f μs \n", micro_seconds * 1000);
    printf("    * %.2f ms \n", micro_seconds);
    printf("    * %.2f s \n", micro_seconds / 1000);
    printf("\n");
}

__global__ void MatrixMulKernelClassic(float* mat_a, float* mat_b, float* res_mat, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < size) && (col < size)) {
        float Pvalue = 0;

        for (int offset = 0; offset < size; offset++) {
            res_mat[row * size + col] += mat_a[row * size + offset] * mat_b[offset * size + col];
        }
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

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        MatrixMulKernelClassic<<<gridDim, blockDim>>>(mat_a_dev, mat_b_dev, mat_res_dev, sizes[i]);
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