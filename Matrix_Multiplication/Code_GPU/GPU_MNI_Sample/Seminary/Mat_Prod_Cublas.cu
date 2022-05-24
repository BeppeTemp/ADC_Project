#include <assert.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define SIZE 4
#define index(i, j, ld) (((j) * (ld)) + (i))

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
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;

    float *mat_a_host = (float *)malloc(SIZE * SIZE * sizeof(float));
    float *mat_b_host = (float *)malloc(SIZE * SIZE * sizeof(float));
    float *mat_a_host_seq = (float *)malloc(SIZE * SIZE * sizeof(float));
    float *mat_b_host_seq = (float *)malloc(SIZE * SIZE * sizeof(float));
    float *mat_res_host = (float *)calloc(SIZE * SIZE, sizeof(float));

    srand(117);
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++) {
            // mat_a_host[index(i, j, SIZE)] = (float)index(i, j, SIZE);
            // mat_b_host[index(i, j, SIZE)] = (float)index(i, j, SIZE);
            mat_a_host[index(i, j, SIZE)] = rand() % 5 - 2;
            mat_b_host[index(i, j, SIZE)] = rand() % 5 - 2;
        }

    srand(117);
    for (int i = 0; i < SIZE * SIZE; i++) {
        mat_a_host[i] = i;
        mat_b_host[i] = i;
        // mat_a_host_seq[i] = rand() % 5 - 2;
        // mat_b_host_seq[i] = rand() % 5 - 2;
    }

    float *mat_a_dev, *mat_b_dev, *mat_res_dev, *mat_res_dev_tras;
    cudaMalloc((void **)&mat_a_dev, SIZE * SIZE * sizeof(float));
    cudaMalloc((void **)&mat_b_dev, SIZE * SIZE * sizeof(float));
    cudaMalloc((void **)&mat_res_dev, SIZE * SIZE * sizeof(float));
    cudaMalloc((void **)&mat_res_dev_tras, SIZE * SIZE * sizeof(float));

    // Setto a_host su a_dev e
    cublasSetMatrix(SIZE, SIZE, sizeof(float), mat_a_host, SIZE, mat_a_dev, SIZE);
    cublasSetMatrix(SIZE, SIZE, sizeof(float), mat_b_host, SIZE, mat_b_dev, SIZE);

    cudaEventRecord(start, 0);

    //  Calcolo il prodotto
    float const alpha(1.0);
    float const beta(0.0);

    cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, SIZE, SIZE, SIZE, &alpha, mat_a_dev, SIZE, mat_b_dev, SIZE, &beta,
                   mat_res_dev, SIZE);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("Tempo GPU=%f\n", elapsed);

    // Calcolo matrice transposta
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, SIZE, SIZE, &alpha, mat_res_dev, SIZE, &beta, mat_res_dev, SIZE,
                mat_res_dev_tras, SIZE);

    cublasGetMatrix(SIZE, SIZE, sizeof(float), mat_res_dev_tras, SIZE, mat_res_host, SIZE);

    printf("\n");
    if (SIZE <= 8) {
        printf("Matrice A:");
        printMat(mat_a_host, SIZE, SIZE);
        printf("Matrice B:");
        printMat(mat_b_host, SIZE, SIZE);
        printf("Matrice Result:");
        printMat(mat_res_host, SIZE, SIZE);
    }

    free(mat_a_host);
    free(mat_b_host);
    free(mat_res_host);
    cudaFree(mat_a_dev);
    cudaFree(mat_b_dev);
    cudaFree(mat_res_dev);

    cublasDestroy_v2(handle);

    return EXIT_SUCCESS;
}