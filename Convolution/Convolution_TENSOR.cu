#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define ARRAY_SIZE 17
#define MASK_SIZE 16

#define PADDED_ARRAY_SIZE (ARRAY_SIZE + (((MASK_SIZE / 2) * 2) - 1))

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Blocco 0 | 0 a 31
// Blocco 1 | 32 0 63
// Blocco 2 | 64 0 95
// Blocco 3 | 96 0 127

#define debug_x 0
#define debug_y 0

using namespace nvcuda;

/*
- La maschera è 16*16
- La matrice 16 * 16 che paddata verrebbe 16+8+8 che per pure caso è 32
*/

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

__global__ void ConvolutionKernelTensor(half* mat, half* mask, float* mat_res) {
    // int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Tile using a 2D grid
    int tile_row = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int tile_col = (blockIdx.y * blockDim.y + threadIdx.y);


    if (threadIdx.x == debug_x && threadIdx.y == debug_y) {
        printf("tile_row: %d\n", tile_row);
        printf("tile_col: %d\n", tile_col);
        printf("\n");
    }

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> mat_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> mask_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    int j = 0;
    for (int col = tile_col * MASK_SIZE; col < MASK_SIZE + (tile_col * MASK_SIZE); col++) {
        for (int row = tile_row * MASK_SIZE; row < MASK_SIZE + (tile_row * MASK_SIZE); row++) {
            // int aRow = i;  // 0
            // int aCol = tile_col * WMMA_M;

            // Bounds checking
            if (row < PADDED_ARRAY_SIZE && col < PADDED_ARRAY_SIZE) {
                if (threadIdx.x == debug_x && threadIdx.y == debug_y) {
                    printf("[%d,%d] Matrice da %d a %d, machera da %d a %d \n", row, col, (row * PADDED_ARRAY_SIZE) + col, (row * PADDED_ARRAY_SIZE) + col + (MASK_SIZE - 1), j, j + (MASK_SIZE - 1));
                }

                // Load the inputs
                wmma::load_matrix_sync(mat_frag, mat + (row * PADDED_ARRAY_SIZE) + col, MASK_SIZE);  // mat_a[aRow, aCol]
                wmma::load_matrix_sync(mask_frag, mask + j, MASK_SIZE);

                // Perform the matrix multiplication
                wmma::mma_sync(acc_frag, mat_frag, mask_frag, acc_frag);
            }
        }
        j = j + MASK_SIZE;
        if (threadIdx.x == debug_x && threadIdx.y == debug_y) {
            printf("\n");
        }
    }

    // int cCol = tile_row * WMMA_N;
    // int cRow = tile_col * WMMA_M;

    // // Store the output
    // wmma::store_matrix_sync(mat_res, acc_frag, PADDED_ARRAY_SIZE, wmma::mem_row_major);  // mat_res[cRow, cCol]
}

int main(void) {
    printf("Size: %d \n", ARRAY_SIZE);
    printf("Mask: %d \n", MASK_SIZE);
    printf("Mask/2: %d \n", MASK_SIZE / 2);
    printf("Padded: %d \n", PADDED_ARRAY_SIZE);
    printf("Centro: [2,2] \n\n");

    half *mat_start_host, *mask_host;
    float* mat_res_host;

    half *mat_start_dev, *mask_dev;
    float* mat_res_dev;

    dim3 gridDim, blockDim;

    // Dichiarazioni
    mat_start_host = (half*)calloc(PADDED_ARRAY_SIZE * PADDED_ARRAY_SIZE, sizeof(half));
    mask_host = (half*)malloc(MASK_SIZE * MASK_SIZE * sizeof(half));
    mat_res_host = (float*)malloc(ARRAY_SIZE * ARRAY_SIZE * sizeof(float));

    cudaMalloc((void**)&mat_start_dev, PADDED_ARRAY_SIZE * PADDED_ARRAY_SIZE * sizeof(half));
    cudaMalloc((void**)&mask_dev, MASK_SIZE * MASK_SIZE * sizeof(half));
    cudaMalloc((void**)&mat_res_dev, ARRAY_SIZE * ARRAY_SIZE * sizeof(float));

    // Inizializzazione Maschera e Matrice iniziale
    for (int i = 0; i < MASK_SIZE * MASK_SIZE; i++) {
        mask_host[i] = __float2half(1);
    }

    for (int i = 0; i < PADDED_ARRAY_SIZE * PADDED_ARRAY_SIZE; i++) {
        mat_start_host[i] = __float2half(i);
    }

    // for (int i = MASK_SIZE / 2; i <= PADDED_ARRAY_SIZE - (MASK_SIZE / 2); i++) {
    //     for (int j = MASK_SIZE / 2; j <= PADDED_ARRAY_SIZE - (MASK_SIZE / 2); j++) {
    //         mat_start_host[i * PADDED_ARRAY_SIZE + j] = __float2half(1);
    //     }
    // }

    //! Debug
    // printf("Mashera: \n");
    // printMat(mask_host, MASK_SIZE, MASK_SIZE);

    // printf("Matrice: \n");
    // printMat(mat_start_host, PADDED_ARRAY_SIZE, PADDED_ARRAY_SIZE);

    // Caricamento in memoria GPU

    cudaMemcpy(mat_start_dev, mat_start_host, PADDED_ARRAY_SIZE * PADDED_ARRAY_SIZE * sizeof(half), cudaMemcpyDefault);
    cudaMemcpy(mask_dev, mask_host, MASK_SIZE * MASK_SIZE * sizeof(half), cudaMemcpyDefault);
    cudaMemset(mat_res_dev, 0, ARRAY_SIZE * ARRAY_SIZE * sizeof(float));

    // Abbiamo dei blocchi da 16 warp che computano ognuno tile di dimensioni 64 * 64
    // crediamo che ogni warp abbia solo 1 tensor che computa una 16*16*16, quindi
    // 16 warp * 16 * 16 * 16
    // supponendo che warp = SM ogni SM ha un Tensor

    //? Ogni 32 thread orizzontali c'è un Tensor e quindi computi 16 elementi

    blockDim.x = 128;
    blockDim.y = 4;
    gridDim.x = (PADDED_ARRAY_SIZE + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    gridDim.y = (PADDED_ARRAY_SIZE + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

    printf("Griglia: %d, %d\n", gridDim.x, gridDim.y);
    printf("Blocco: %d, %d\n", blockDim.x, blockDim.y);
    printf("\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    ConvolutionKernelTensor<<<gridDim, blockDim>>>(mat_start_dev, mask_dev, mat_res_dev);
    cudaEventRecord(stop);

    cudaMemcpy(mat_res_host, mat_res_dev, ARRAY_SIZE * ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // printf("Matrice risultate:\n");
    // printMat(mat_res_host, ARRAY_SIZE, ARRAY_SIZE);

    free(mat_start_host);
    free(mask_host);
    free(mat_res_host);

    cudaFree(mask_dev);
    cudaFree(mat_res_dev);

    return 0;
}