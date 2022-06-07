#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <chrono>

#define SIZE 4194304
#define BLOCK_DIM 1

using namespace std::chrono;

// Somma Seriale
__host__ void CPU_SUM(float* a, float* b, float* c, int n) {
    int i;
    for (i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

// Somma Parallela
__global__ void GPU_SUM(float* a, float* b, float* c, int n) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    // Verifica che non ci sia un overhead in modo da non far lavorare i
    // thread su aree di memoria invalide. (Esempio 9 elementi su blocchi da 4)
    if (index < n)
        c[index] = a[index] + b[index];
}

// Stampa dei tempi
void time_stats(float micro_seconds) {
    printf("Execution times:\n");
    printf("    * %.0f μs \n", micro_seconds);
    printf("    * %.2f ms \n", micro_seconds / 1000);
    printf("    * %.2f s \n", micro_seconds / 1000 / 1000);
    printf("    * %.0f minutes and %d seconds\n",
           ((micro_seconds / 1000) / 1000) / 60,
           (int)((micro_seconds / 1000) / 1000) % 60);
}

// Stampa vettore
void printVet(float* vet, int size) {
    printf("\n");
    for (int i = 0; i < size; i++) {
        printf("|%.2f", vet[i]);
    }
    printf("|\n\n");
}

int main(void) {
    float *a_host, *b_host, *res_host_gpu, *res_host_cpu;  // host data
    float *a_dev, *b_dev, *c_dev;                          // device data

    // determinazione esatta del numero di blocchi
    dim3 gridDim, blockDim;
    blockDim.x = BLOCK_DIM;
    gridDim.x = SIZE / blockDim.x + ((SIZE % blockDim.x) == 0 ? 0 : 1);

    // Allocazione memoria
    int nBytes = SIZE * sizeof(float);

    a_host = (float*)malloc(nBytes);
    b_host = (float*)malloc(nBytes);
    res_host_gpu = (float*)malloc(nBytes);
    res_host_cpu = (float*)malloc(nBytes);

    cudaMalloc((void**)&a_dev, nBytes);
    cudaMalloc((void**)&b_dev, nBytes);
    cudaMalloc((void**)&c_dev, nBytes);

    // inizializzo i dati
    srand((unsigned int)time(0));

    for (int i = 0; i < SIZE; i++) {
        a_host[i] = 1;
        b_host[i] = 2;
    }

    cudaMemcpy(a_dev, a_host, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b_host, nBytes, cudaMemcpyHostToDevice);

    // azzeriamo il contenuto del vettore c
    memset(res_host_gpu, 0, nBytes);
    cudaMemset(c_dev, 0, nBytes);

    CPU_SUM(a_host, b_host, res_host_cpu, SIZE);

    auto start = high_resolution_clock::now();

    // invocazione del kernel
    GPU_SUM<<<gridDim, blockDim>>>(a_dev, b_dev, c_dev, SIZE);

    cudaDeviceSynchronize();

    auto stop = high_resolution_clock::now();

    cudaMemcpy(res_host_gpu, c_dev, nBytes, cudaMemcpyDeviceToHost);

    bool flag = false;

    for (int i = 0; i < SIZE; i++) {
        if (res_host_cpu[i] != res_host_gpu[i]) {
            flag = true;
        }
    }

    if (flag) {
        printf("TUTT SBAGLIAT\n");

    } else {
        printf("TUTT APPOST\n");
        time_stats(duration_cast<microseconds>(stop - start).count());
    }

    free(a_host);
    free(b_host);
    free(res_host_gpu);

    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);

    return 0;
}