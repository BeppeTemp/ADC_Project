#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>

#define SIZE 4

#define BLOCK_DIM 8

// Calcolo prodotto matriciale
__global__ void MatrixMulKernel(float *mat_a, float *mat_b, float *res_mat) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((Row < SIZE) && (Col < SIZE)) {
        float Pvalue = 0;

        for (int i = 0; i < SIZE; ++i) {
            Pvalue += mat_a[Row * SIZE + i] * mat_b[i * SIZE + Col];
        }
        res_mat[Row * SIZE + Col] = Pvalue;
    }
}

// Stampa della matrice
void printMat(float *mat, int row, int column) {
    printf("\n");
    for (int i = 0; i < (row * column); i++) {
        printf("|");
        printf("%06.2f", mat[i]);
        if (((i + 1) % (column) == 0) && (i != 0)) printf("|\n");
        if ((row * column) == 1) printf("|\n");
        if (column == 1 && ((i == 0))) printf("|\n");
    }
    printf("\n");
}

int main(void) {
    float *mat_a_host, *mat_b_host, *mat_res_host_gpu;
    float *mat_a_dev, *mat_b_dev, *mat_res_dev;
    dim3 gridDim, blockDim;

    // Allocazione memoria
    int nBytes = SIZE * SIZE * sizeof(float);

    mat_a_host = (float *)malloc(nBytes);
    mat_b_host = (float *)malloc(nBytes);
    mat_res_host_gpu = (float *)calloc(nBytes, sizeof(float));

    cudaMalloc((void **)&mat_a_dev, nBytes);
    cudaMalloc((void **)&mat_b_dev, nBytes);
    cudaMalloc((void **)&mat_res_dev, nBytes);

    srand(117);
    for (int i = 0; i < SIZE * SIZE; i++) {
        mat_a_host[i] = i;
        mat_b_host[i] = i;
        // mat_a_host[i] = rand() % 5 - 2;
        // mat_b_host[i] = rand() % 5 - 2;
    }

    // Copia dei dati dall'host al device
    cudaMemcpy(mat_a_dev, mat_a_host, nBytes, cudaMemcpyDefault);
    cudaMemcpy(mat_b_dev, mat_b_host, nBytes, cudaMemcpyDefault);
    cudaMemset(mat_res_dev, 0, nBytes);

    // Configurazione del kernel
    blockDim.x = BLOCK_DIM;
    blockDim.y = BLOCK_DIM;
    gridDim.x = SIZE / blockDim.x + ((SIZE % blockDim.x) == 0 ? 0 : 1);
    gridDim.y = SIZE / blockDim.y + ((SIZE % blockDim.y) == 0 ? 0 : 1);

    // Creazione eventi
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Inizio computazione su GPU
    cudaEventRecord(start);
    MatrixMulKernel<<<gridDim, blockDim>>>(mat_a_dev, mat_b_dev, mat_res_dev);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("Tempo GPU=%f\n", elapsed);

    cudaMemcpy(mat_res_host_gpu, mat_res_dev, nBytes, cudaMemcpyDeviceToHost);

    printf("\n");
    if (SIZE <= 8) {
        printf("Matrice A:");
        printMat(mat_a_host, SIZE, SIZE);
        printf("Matrice B:");
        printMat(mat_b_host, SIZE, SIZE);
        printf("Matrice result della GPU:");
        printMat(mat_res_host_gpu, SIZE, SIZE);
    }

    free(mat_a_host);
    free(mat_b_host);
    free(mat_res_host_gpu);

    cudaFree(mat_a_dev);
    cudaFree(mat_b_dev);
    cudaFree(mat_res_dev);

    return 0;
}