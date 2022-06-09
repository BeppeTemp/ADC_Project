#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

#define PRINT_GREEN(str) printf("\x1b[32m%s\x1b[0m", str);
#define PRINT_RED(str) printf("\x1b[31m%s\x1b[0m", str);

#define BLOCK_DIM 32
#define TILE_WIDTH 32

void time_stats(float micro_seconds) {
    printf("Execution times:\n");
    printf("    * %.0f μs \n", micro_seconds * 1000);
    printf("    * %.2f ms \n", micro_seconds);
    printf("    * %.2f s \n", micro_seconds / 1000);
    printf("\n");
}

__global__ void MatrixMulKernelTiled(float* mat_a, float* mat_b, float* res_mat, int size) {
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
        __shared__ float m_a_sh[BLOCK_DIM][BLOCK_DIM];
        __shared__ float m_b_sh[BLOCK_DIM][BLOCK_DIM];

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
        MatrixMulKernelTiled<<<gridDim, blockDim>>>(mat_a_dev, mat_b_dev, mat_res_dev, sizes[i]);
        cudaEventRecord(stop);

        cudaMemcpy(mat_res_host_gpu, mat_res_dev, nBytes, cudaMemcpyDeviceToHost);

        bool check = true;
        for (int k = 0; k < sizes[i] * sizes[i]; k++) {
            if (mat_res_host_gpu[i] != sizes[i])
                check = false;
        }

        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);

        printf("Matrix size: %d x %d \n", sizes[i], sizes[i]);
        printf("Block size: %d x %d = %d\n", BLOCK_DIM, BLOCK_DIM, BLOCK_DIM * BLOCK_DIM);
        printf("TFLOPS: %.2f\n", static_cast<double>((static_cast<double>(sizes[i]) * sizes[i] * sizes[i] * 2) / (elapsed / 1000.)) / 1e12);
        printf("Check: ");
        if (check) {
            PRINT_GREEN("Verified\n");
        } else {
            PRINT_RED("Error\n");
        }
        time_stats(elapsed);

        free(mat_a_host);
        free(mat_b_host);

        cudaFree(mat_a_dev);
        cudaFree(mat_b_dev);
        cudaFree(mat_res_dev);
    }

    return 0;
}