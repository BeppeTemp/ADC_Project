#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define ARRAY_SIZE 6
#define MASK_SIZE 4

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

__global__ void ConvolutionKernelTensor(half* unfold_mat, half* mask, float* mat_c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Tile using a 2D grid
    int tile_row = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int tile_col = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> unf_row_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> mask_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // I cicli qua dentro per qualche motivo salgono solo di 16 per volta
    for (int i = 0; i < UNF_ARRAY_M; i += WMMA_N) {
        int aCol = i;  // 0
        int aRow = tile_row * WMMA_M;

        // Load the inputs
        wmma::load_matrix_sync(unf_row_frag, unfold_mat + (aRow * UNF_ARRAY_N) + aCol, MASK_SIZE);
        wmma::load_matrix_sync(mask_frag, mask, MASK_SIZE * MASK_SIZE);

        // Perform the matrix multiplication
        wmma::mma_sync(acc_frag, unf_row_frag, mask_frag, acc_frag);
    }

    int cCol = tile_row * WMMA_N;
    int cRow = tile_col * WMMA_M;

    // Store the output
    wmma::store_matrix_sync(mat_c + (cRow * UNF_ARRAY_N) + cCol, acc_frag, UNF_ARRAY_N, wmma::mem_row_major);  // mat_c[cRow, cCol]
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
    // printMat(mask_host, MASK_SIZE, MASK_SIZE);

    printf("Matrice: \n");
    // printMat(mat_start_host, PADDED_ARRAY_SIZE, PADDED_ARRAY_SIZE);

    matrixUnfold(mat_start_host, mat_unfolded_host);

    printf("Matrice finale unfolded:\n");
    // printMat(mat_unfolded_host, UNF_ARRAY_M, UNF_ARRAY_N);

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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    ConvolutionKernelTensor<<<gridDim, blockDim>>>(mat_unfolded_dev, mask_dev, mat_res_dev);
    cudaEventRecord(stop);

    cudaMemcpy(mat_res_host, mat_res_dev, ARRAY_SIZE * ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // printf("Matrice risultate:\n");
    // printMat(mat_res_host, 16, 16);

    free(mat_start_host);
    free(mask_host);
    free(mat_res_host);

    cudaFree(mask_dev);
    cudaFree(mat_res_dev);

    return 0;
}