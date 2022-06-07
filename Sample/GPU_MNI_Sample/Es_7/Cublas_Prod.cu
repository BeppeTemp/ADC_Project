#include <cublas_v2.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define SIZE 4

int main(void) {
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;

    float *a_host, *b_host;
    float *a_dev, *b_dev;
    float res = 0;

    a_host = (float *)malloc(SIZE * sizeof(*a_host));
    b_host = (float *)malloc(SIZE * sizeof(*b_host));

    if (!b_host || !a_host) {
        printf("host memory allocation failed");
        return EXIT_FAILURE;
    }

    // Inizializzo i dati
    srand((unsigned int)time(0));
    for (int i = 0; i < SIZE; i++) {
        a_host[i] = rand() % 5 - 2;
        b_host[i] = rand() % 5 - 2;
    }

    // Alloco a_dev e b_dev
    cudaStat = cudaMalloc((void **)&a_dev, SIZE * sizeof(*a_host));
    if (cudaStat != cudaSuccess) {
        printf("device memory allocation failed");
        return EXIT_FAILURE;
    }
    cudaStat = cudaMalloc((void **)&b_dev, SIZE * sizeof(*b_host));
    if (cudaStat != cudaSuccess) {
        printf("device memory allocation failed");
        return EXIT_FAILURE;
    }

    // Creo l'handle per cublas
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    // Setto a_host su a_dev
    stat = cublasSetVector(SIZE, sizeof(float), a_host, 1, a_dev, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("data download failed");
        cudaFree(a_dev);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    // Setto b_host su b_dev
    stat = cublasSetVector(SIZE, sizeof(float), b_host, 1, b_dev, 1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("data download failed");
        cudaFree(b_dev);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    cudaEventRecord(start, 0);

    // Calcolo il prodotto
    stat = cublasSdot(handle, SIZE, a_dev, 1, b_dev, 1, &res);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("data download failed cublasSdot");
        cudaFree(a_dev);
        cudaFree(b_dev);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);

    printf("Tempo GPU=%f\n\n", elapsed);
    printf("Risultato del prodotto: %f\n", res);

    cudaFree(a_dev);
    cudaFree(b_dev);
    cublasDestroy(handle);
    free(a_host);
    free(b_host);

    return EXIT_SUCCESS;
}