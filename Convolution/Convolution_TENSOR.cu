#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define ARRAY_SIZE 5
#define MASK_SIZE 3

#define PADDED_ARRAY_SIZE (ARRAY_SIZE + ((MASK_SIZE / 2) * 2))

#define UNF_ARRAY_M ((PADDED_ARRAY_SIZE - MASK_SIZE + 1) * (PADDED_ARRAY_SIZE - MASK_SIZE + 1))
#define UNF_ARRAY_N (MASK_SIZE * MASK_SIZE)

#define STEP_X (PADDED_ARRAY_SIZE - MASK_SIZE)
#define STEP_Y (PADDED_ARRAY_SIZE - MASK_SIZE)

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

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
        printf(" %01.0f ", __half2float(mat[i]));
        if (((i + 1) % (n) == 0) && (i != 0))
            printf("|\n");
        if ((m * n) == 1)
            printf("|\n");
        if (n == 1 && ((i == 0)))
            printf("|\n");
    }
    printf("\n");
}

void matrixUnfold(half* mat_start, half* mat_unfolded) {
    int step_x = STEP_X;
    int step_y = STEP_Y;

    int k = 0;
    for (int t = 0; t < PADDED_ARRAY_SIZE - MASK_SIZE + 1; t++) {
        for (int h = 0; h < PADDED_ARRAY_SIZE - MASK_SIZE + 1; h++) {
            // printf("Stampo da [%d, %d] a [%d,%d]\n", t, h, PADDED_ARRAY_SIZE - step_x - 1, PADDED_ARRAY_SIZE - step_y - 1);
            for (int i = t; i < PADDED_ARRAY_SIZE - step_x; i++) {
                for (int j = h; j < PADDED_ARRAY_SIZE - step_y; j++) {
                    // printf("[%d,%d] ", i, j);
                    // printf("%01.0f ", mat_start[i * PADDED_ARRAY_SIZE + j]);
                    mat_unfolded[k] = mat_start[i * PADDED_ARRAY_SIZE + j];
                    k++;
                }
            }
            step_y--;
            // printf("\n");
        }
        step_y = STEP_X;
        step_x--;
    }
}

__global__ void ConvolutionKernelTensor(half* mat_a, half* mask, float* mat_c) {
    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    // int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Tile using a 2D grid
    int tile_row = (blockIdx.x * blockDim.x + threadIdx.x) / 16;
    int tile_col = (blockIdx.y * blockDim.y + threadIdx.y);

    // if (threadIdx.x < SIZE * SIZE / blockDim.y)
    //     printf("Block_Dim: [%d,%d], Block_Thread: [%d,%d], Coord_Thread: [%d,%d], Tile M/N: [%d,%d]\n", blockDim.x, blockDim.y, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, tile_row, tile_col);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> mask_frag;
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
            wmma::load_matrix_sync(mask_frag, mask + (bRow * 16) + bCol, 16);    // mask[bRow, bCol]

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, mask_frag, acc_frag);
        }
    }

    int cCol = tile_row * WMMA_N;
    int cRow = tile_col * WMMA_M;

    // Store the output
    wmma::store_matrix_sync(mat_c + (cRow * size) + cCol, acc_frag, size, wmma::mem_row_major);  // mat_c[cRow, cCol]
}

int main(void) {
    printf("Size: %d \n", ARRAY_SIZE);
    printf("Mask: %d \n", MASK_SIZE);
    printf("Mask/2: %d \n", MASK_SIZE / 2);
    printf("Padded: %d \n", PADDED_ARRAY_SIZE);
    printf("Unf_Array_M: %d \n", UNF_ARRAY_M);
    printf("Unf_Array_N: %d \n", UNF_ARRAY_N);
    printf("Step_X: %d \n", STEP_X);
    printf("Step_Y: %d \n\n", STEP_Y);

    half *mat_start_host, *mat_unfolded_host, *mask_host;
    float* mat_res_host;

    half *mat_unfolded_dev, *mask_dev;
    float* mat_res_dev;

    dim3 gridDim, blockDim;

    mat_start_host = (half*)calloc(PADDED_ARRAY_SIZE * PADDED_ARRAY_SIZE, sizeof(half));
    mat_unfolded_host = (half*)calloc(UNF_ARRAY_M * UNF_ARRAY_N, sizeof(half));
    mask_host = (half*)malloc(MASK_SIZE * MASK_SIZE * sizeof(half));
    mat_res_host = (float*)malloc(ARRAY_SIZE * ARRAY_SIZE * sizeof(float));

    cudaMalloc((void**)&mat_unfolded_dev, UNF_ARRAY_M * UNF_ARRAY_N * sizeof(half));
    cudaMalloc((void**)&mask_dev, MASK_SIZE * MASK_SIZE * sizeof(half));
    cudaMalloc((void**)&mat_res_dev, ARRAY_SIZE * ARRAY_SIZE * sizeof(float));

    // Inizializzazione Maschera e Matrice iniziale
    for (int i = 0; i < MASK_SIZE * MASK_SIZE; i++) {
        mask_host[i] = __float2half(1);
    }

    for (int i = MASK_SIZE / 2; i < PADDED_ARRAY_SIZE - (MASK_SIZE / 2); i++) {
        for (int j = MASK_SIZE / 2; j < PADDED_ARRAY_SIZE - (MASK_SIZE / 2); j++) {
            mat_start_host[i * PADDED_ARRAY_SIZE + j] = __float2half(1);
        }
    }

    printf("Mashera: \n");
    printMat(mask_host, MASK_SIZE, MASK_SIZE);

    printf("Matrice: \n");
    printMat(mat_start_host, PADDED_ARRAY_SIZE, PADDED_ARRAY_SIZE);

    matrixUnfold(mat_start_host, mat_unfolded_host);

    printf("Matrice finale unfolded:\n");
    printMat(mat_unfolded_host, UNF_ARRAY_M, UNF_ARRAY_N);

    // Caricamento in memoria GPU

    cudaMemcpy(mat_unfolded_dev, mat_unfolded_host, UNF_ARRAY_M * UNF_ARRAY_N * sizeof(half), cudaMemcpyDefault);
    cudaMemcpy(mask_dev, mask_host, MASK_SIZE * MASK_SIZE * sizeof(half), cudaMemcpyDefault);
    cudaMemset(mat_res_dev, 0, ARRAY_SIZE * ARRAY_SIZE * sizeof(float));

    // Abbiamo dei blocchi da 16 warp che computano ognuno tile di dimensioni 64 * 64
    // crediamo che ogni warp abbia solo 1 tensor che computa una 16*16*16, quindi
    // 16 warp * 16 * 16 * 16
    // supponendo che warp = SM ogni SM ha un Tensor

    blockDim.x = 128;
    blockDim.y = 4;
    gridDim.x = (UNF_ARRAY_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    gridDim.y = (UNF_ARRAY_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

    printf("Griglia: %d, %d\n", gridDim.x, gridDim.y);
    printf("Blocco: %d, %d\n", blockDim.x, blockDim.y);
    printf("\n");

    ConvolutionKernelTensor<<<gridDim, blockDim>>>(mat_unfolded_dev, mask_dev, mat_res_dev);

    // cudaMemcpy(mat_res_host, mat_res_dev, 16 * 16 * sizeof(float), cudaMemcpyDeviceToHost);

    // // printf("Matrice risultate:\n");
    // // printMat(mat_res_host, 16, 16);

    // free(mat_start_host);
    // free(mask_host);
    // free(mat_res_host);

    // cudaFree(mat_start_dev);
    // cudaFree(mask_dev);
    // cudaFree(mat_res_dev);

    return 0;
}