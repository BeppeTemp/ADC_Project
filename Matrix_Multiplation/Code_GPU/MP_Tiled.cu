#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <time.h>

#define SIZE 4

// Blocchi 8 x 8  grandezza ottimale
#define BLOCK_DIM 8
#define TILE_WIDTH 8

// Calcolo prodotto matriciale
__global__ void MatrixMulKernel(float *mat_a, float *mat_b, float *res_mat) {
    __shared__ float M_ds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float N_ds[TILE_WIDTH][TILE_WIDTH];

    // Salvataggio delle coordinate all'interno di registri per aumentare le performance
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Identify the row and column of the d_P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    if ((Row < SIZE) && (Col < SIZE)) {
        float Pvalue = 0;
        // Loop over the d_M and d_N tiles required to compute d_P element
        for (int m = 0; m <= SIZE / TILE_WIDTH; ++m) {
            // Coolaborative loading of d_M and d_N tiles into shared memory
            M_ds[ty][tx] = mat_a[Row * SIZE + m * TILE_WIDTH + tx];
            N_ds[ty][tx] = mat_b[(m * TILE_WIDTH + ty) * SIZE + Col];

            __syncthreads();

            for (int k = 0; k < TILE_WIDTH; ++k) {
                Pvalue += M_ds[ty][k] * N_ds[k][tx];
            }
            __syncthreads();
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

    srand(time(NULL));
    for (int i = 0; i < SIZE * SIZE; i++) {
        mat_a_host[i] = i;
        mat_b_host[i] = i;
        // a_host[i] = rand() % 5 - 2;
        // b_host[i] = rand() % 5 - 2;
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
        printf("Vettore result della GPU:");
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