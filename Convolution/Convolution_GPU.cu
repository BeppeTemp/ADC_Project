#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <chrono>

#define MASK_SIZE 4
#define MASK_CENTER 1

#define BLOCK_DIM 32
#define TILE_WIDTH 32

using namespace std::chrono;

void time_stats(float micro_seconds) {
    printf("Execution times:\n");
    printf("    * %.0f μs \n", micro_seconds * 1000);
    printf("    * %.2f ms \n", micro_seconds);
    printf("    * %.2f s \n", micro_seconds / 1000);
    printf("\n");
}

__global__ void ConvolutionKernel(float* mat_start, float* mat_res, const float* Mask, int size) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Coordinate result
    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;

    // Corrdinate start
    int row_i = row_o - MASK_SIZE / 2;
    int col_i = col_o - MASK_SIZE / 2;

    __shared__ float N_ds[TILE_WIDTH + MASK_SIZE - 1][TILE_WIDTH + MASK_SIZE - 1];

    if ((row_i >= 0) && (row_i < size) && (col_i >= 0) && (col_i < size)) {
        N_ds[ty][tx] = data[row_i * size + col_i];
    }

    float output = 0.0f;
    if (ty < TILE_WIDTH && tx < TILE_WIDTH) {
        for (int i = 0; i < MASK_SIZE; i++) {
            for (int j = 0; j < MASK_SIZE; j++) {
                output += Mask[i][j] * N_ds[i + ty][j + tx];
            }
        }
        if (row_o < size && col_o < size) {
            mat_res[row_o * size + col_o] = output;
        }
    }
}

int main() {
    int sizes[5] = {2048, 4096, 8192, 16384, 32768};

    float *mat_start_host, *mat_res_host, *mask_host;
    float *mat_start_dev, *mat_res_dev, *mask_dev;

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

        // blockDim.x = BLOCK_DIM;
        // blockDim.y = BLOCK_DIM;
        // gridDim.x = sizes[i] / blockDim.x + ((sizes[i] % blockDim.x) == 0 ? 0 : 1);
        // gridDim.y = sizes[i] / blockDim.y + ((sizes[i] % blockDim.y) == 0 ? 0 : 1);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        // ConvolutionKernel<<<gridDim, blockDim>>>();
        cudaEventRecord(stop);

        cudaMemcpy(mat_res_host, mat_res_dev, sizes[k] * sizes[k] * sizeof(float), cudaMemcpyDeviceToHost);

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

    free(mat_res_host);
    return 0;
}