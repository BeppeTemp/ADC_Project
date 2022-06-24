#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

#define ARRAY_SIZE 16
#define MASK_SIZE 4

#define PADDED_ARRAY_SIZE (ARRAY_SIZE + (((MASK_SIZE / 2) * 2)))

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define debug_x 32
#define debug_y 0

using namespace nvcuda;

void printMat(half* mat, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        printf("|");
        printf(" %02.0f ", __half2float(mat[i]));
        if (((i + 1) % (n) == 0) && (i != 0))
            printf("|\n");
        if ((m * n) == 1)
            printf("|\n");
        if (n == 1 && ((i == 0)))
            printf("|\n");
    }
    printf("\n");
}

void printMat(float* mat, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        printf("|");
        printf(" %02.0f ", mat[i]);
        if (((i + 1) % (n) == 0) && (i != 0))
            printf("|\n");
        if ((m * n) == 1)
            printf("|\n");
        if (n == 1 && ((i == 0)))
            printf("|\n");
    }
    printf("\n");
}

// Do transpose of a matrix
void mat_traspose(half* mat) {
    for (int i = 0; i < MASK_SIZE / 2; i++) {
        for (int j = 0; j < MASK_SIZE / 2; j++) {
            mat[i * WMMA_N + j] = mat[j * WMMA_M + i];
        }
    }
}

__global__ void WMMAF16TensorCore(half* mat, half* mask, half* mat_temp, float* mat_res) {
    int tile_row = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int tile_col = (blockIdx.y * blockDim.y + threadIdx.y);

    if (threadIdx.x == debug_x && threadIdx.y == debug_y) {
        printf("tile_row: %d\n", tile_row);
        printf("tile_col: %d\n", tile_col);
        printf("\n");
    }

    if (threadIdx.x == debug_x && threadIdx.y == debug_y) {
        //         printf("tot: %d\n", tot);
        if (threadIdx.x == debug_x && threadIdx.y == debug_y) {
            for (int i = 0; i < PADDED_ARRAY_SIZE * PADDED_ARRAY_SIZE; i++) {
                printf("|");
                printf(" %02.0f ", __half2float(mat[i]));
                if (((i + 1) % (PADDED_ARRAY_SIZE) == 0) && (i != 0))
                    printf("|\n");
                if ((PADDED_ARRAY_SIZE * PADDED_ARRAY_SIZE) == 1)
                    printf("|\n");
                if (PADDED_ARRAY_SIZE == 1 && ((i == 0)))
                    printf("|\n");
            }
            printf("\n");
        }
    }

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> mat_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> mask_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    // Charge mask fragment
    wmma::load_matrix_sync(mask_frag, mask, MASK_SIZE);  // mask[bRow, bCol]

    // for (int col = tile_col * PADDED_ARRAY_SIZE / 2; col < MASK_SIZE + (tile_col * PADDED_ARRAY_SIZE / 2); col++) {
    //     for (int row = tile_row * PADDED_ARRAY_SIZE / 2; row < MASK_SIZE + (tile_row * PADDED_ARRAY_SIZE / 2); row++) {
    for (int row = 0; row < 16; row++) {
        for (int col = 0; col < 32; col = col + 2) {
            if (row < ARRAY_SIZE && col < ARRAY_SIZE * 2) {
                // Load the inputs
                wmma::load_matrix_sync(mat_frag, mat + ((row * PADDED_ARRAY_SIZE) + col + 2), 16);

                // Perform the matrix multiplication
                wmma::mma_sync(acc_frag, mat_frag, mask_frag, acc_frag);

                // Store the output
                wmma::store_matrix_sync(mat_temp, acc_frag, 16, wmma::mem_row_major);
                //     // wmma::store_matrix_sync(mat_temp + (aRow * size) + aCol, acc_frag, size, wmma::mem_col_major);

                half diag = 0;
                for (int i = 0; i < MASK_SIZE; i++) {
                    diag += mat_temp[i * MASK_SIZE + i];
                }

                mat_res[(row * ARRAY_SIZE) + (col / 2)] = __float2half(diag);

                if (threadIdx.x == debug_x && threadIdx.y == debug_y) {
                    printf("Coordinate A [%d,%d]\n", row, col);
                    printf("Coordinate Scarico [%d,%d]\n", row, col / 2);
                    printf("tot: %02.0f\n", __half2float(diag));
                    //     if (threadIdx.x == debug_x && threadIdx.y == debug_y) {
                    //         for (int i = 0; i < MASK_SIZE * MASK_SIZE; i++) {
                    //             printf("|");
                    //             printf(" %06.1f ", __half2float(mat[i]));
                    //             if (((i + 1) % (MASK_SIZE) == 0) && (i != 0))
                    //                 printf("|\n");
                    //             if ((MASK_SIZE * MASK_SIZE) == 1)
                    //                 printf("|\n");
                    //             if (MASK_SIZE == 1 && ((i == 0)))
                    //                 printf("|\n");
                    //         }
                    //         printf("\n");
                    //     }
                }
            }
        }
    }
}

int main(void) {
    half *mat_host, *mask_host;
    float* mat_res_host_gpu;

    half *mat_dev, *mask_dev, *mat_temp_dev;
    float* mat_res_dev;

    dim3 gridDim, blockDim;

    mat_host = (half*)calloc(PADDED_ARRAY_SIZE * PADDED_ARRAY_SIZE, sizeof(half));
    mask_host = (half*)malloc(MASK_SIZE * MASK_SIZE * sizeof(half));
    mat_res_host_gpu = (float*)malloc(ARRAY_SIZE * ARRAY_SIZE * sizeof(float));

    cudaMalloc((void**)&mat_dev, PADDED_ARRAY_SIZE * PADDED_ARRAY_SIZE * sizeof(half));
    cudaMalloc((void**)&mask_dev, MASK_SIZE * MASK_SIZE * sizeof(half));
    cudaMalloc((void**)&mat_temp_dev, MASK_SIZE * MASK_SIZE * sizeof(half));
    cudaMalloc((void**)&mat_res_dev, ARRAY_SIZE * ARRAY_SIZE * sizeof(float));

    float k = 0;
    for (int i = MASK_SIZE / 2; i <= PADDED_ARRAY_SIZE - (MASK_SIZE / 2); i++) {
        for (int j = MASK_SIZE / 2; j <= PADDED_ARRAY_SIZE - (MASK_SIZE / 2); j++) {
            mat_host[i * PADDED_ARRAY_SIZE + j] = __float2half(k);
            k += 0.005;
        }
    }

    k = 0;
    for (int j = 0; j < MASK_SIZE * MASK_SIZE; j++) {
        mask_host[j] = __float2half(k);
        k += 0.005;
    }

    // printf("Mat: \n");
    // printMat(mat_host, PADDED_ARRAY_SIZE, PADDED_ARRAY_SIZE);

    printf("Maschera: \n");
    printMat(mask_host, MASK_SIZE, MASK_SIZE);

    mat_traspose(mask_host);

    printf("Maschera Trans: \n");
    printMat(mask_host, MASK_SIZE, MASK_SIZE);


    cudaMemcpy(mat_dev, mat_host, PADDED_ARRAY_SIZE * PADDED_ARRAY_SIZE * sizeof(half), cudaMemcpyDefault);
    cudaMemcpy(mask_dev, mask_host, MASK_SIZE * MASK_SIZE * sizeof(half), cudaMemcpyDefault);
    cudaMemset(mat_temp_dev, 0, MASK_SIZE * MASK_SIZE * sizeof(half));
    cudaMemset(mat_res_dev, 0, ARRAY_SIZE * ARRAY_SIZE * sizeof(half));

    blockDim.x = 128;
    blockDim.y = 4;
    gridDim.x = (ARRAY_SIZE + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    gridDim.y = (ARRAY_SIZE + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

    printf("Padded: %d\n", PADDED_ARRAY_SIZE);
    printf("Griglia: %d, %d\n", gridDim.x, gridDim.y);
    printf("Blocco: %d, %d\n", blockDim.x, blockDim.y);
    printf("\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    WMMAF16TensorCore<<<gridDim, blockDim>>>(mat_dev, mask_dev, mat_temp_dev, mat_res_dev);
    cudaEventRecord(stop);

    cudaMemcpy(mat_res_host_gpu, mat_res_dev, ARRAY_SIZE * ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // printf("Matrix size: %d x %d \n", sizes[i], sizes[i]);
    // printf("Check: ");
    // if (check) {
    //     PRINT_GREEN("Verified\n");
    // } else {
    //     PRINT_RED("Error\n");
    // }

    // printf("\nRisultato:\n");
    // printMat(mat_res_host_gpu, ARRAY_SIZE, ARRAY_SIZE);

    free(mat_host);
    free(mask_host);

    cudaFree(mat_dev);
    cudaFree(mask_dev);
    cudaFree(mat_res_dev);

    return 0;
}