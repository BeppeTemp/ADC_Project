#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

#define PRINT_GREEN(str) printf("\x1b[32m%s\x1b[0m", str);
#define PRINT_RED(str) printf("\x1b[31m%s\x1b[0m", str);

// MMA matrix tile dimensions.
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

__global__ void WMMAF16TensorCore(half* mat_a, half* mat_b, float* mat_c, int size) {
    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    // int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Tile using a 2D grid
    int tile_row = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int tile_col = (blockIdx.y * blockDim.y + threadIdx.y);


    
    printf("Block_Dim: [%d,%d], Block_Thread: [%d,%d], Coord_Thread: [%d,%d], Tile M/N: [%d,%d]\n", blockDim.x, blockDim.y, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, tile_row, tile_col);
     

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
            // printf("Block_Thread: [%d,%d], Coord_Thread: [%d,%d], A[%d,%d], B[%d,%d], ID: %ld, IDM: %ld, I:%d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, aRow, aCol, bRow, bCol, mat_a, i);
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
    wmma::store_matrix_sync(mat_c + (cRow * size) + cCol, acc_frag, size, wmma::mem_row_major);  // mat_c[cRow, cCol]
}

int main(void) {
    int sizes[1] = {32};

    half *mat_a_host, *mat_b_host;
    float* mat_res_host_gpu;
    half *mat_a_dev, *mat_b_dev;
    float* mat_res_dev;
    dim3 gridDim, blockDim;

    for (int i = 0; i < 1; i++) {
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

        printf("Griglia: %d, %d\n", gridDim.x, gridDim.y);
        printf("Blocco: %d, %d\n", blockDim.x, blockDim.y);
        printf("\n");

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        WMMAF16TensorCore<<<gridDim, blockDim>>>(mat_a_dev, mat_b_dev, mat_res_dev, sizes[i]);
        cudaEventRecord(stop);

        cudaMemcpy(mat_res_host_gpu, mat_res_dev, nBytes, cudaMemcpyDeviceToHost);

        bool check = true;
#pragma omp parallel for
        for (int k = 0; k < sizes[i] * sizes[i]; k++) {
            if (mat_res_host_gpu[i] != sizes[i])
                check = false;
        }

        printf("Matrix size: %d x %d \n", sizes[i], sizes[i]);
        printf("Check: ");
        if (check) {
            PRINT_GREEN("Verified\n");
        } else {
            PRINT_RED("Error\n");
        }

        //  printMat(mat_res_host_gpu, sizes[i]);
        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        time_stats(elapsed);
        printf("TFLOPS: %.2f\n", static_cast<double>((static_cast<double>(sizes[i]) * sizes[i] * sizes[i] * 2) / (elapsed / 1000.)) / 1e12);

        free(mat_a_host);
        free(mat_b_host);

        cudaFree(mat_a_dev);
        cudaFree(mat_b_dev);
        cudaFree(mat_res_dev);
    }

    return 0;
}