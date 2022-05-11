#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <time.h>

#define SIZE 10
#define BLOCK_DIM 2

// Somma Seriale
__host__ void CPU_SUM(float *a, float *b, float *c, int n) {
    int i;
    for (i = 0; i < n; i++) c[i] = a[i] + b[i];
}
// Somma Parallela
__global__ void GPU_SUM(float *a, float *b, float *c, int n) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    // Verifica che non ci sia un overhead in modo da non far lavorare i
    // thread su aree di memoria invalide. (Esempio 9 elementi su blocchi da 4)
    if (index < n) c[index] = a[index] + b[index];
}
// Stampa vettore 
void printVet(float *vet, int size) {
    printf("\n");
    for (int i = 0; i < size; i++) {
        printf("|%.2f", vet[i]);
    }
    printf("|\n\n");
}

int main(void) {
    float *a_host, *b_host, *res_host_gpu, *res_host_cpu;  // host data
    float *a_dev, *b_dev, *c_dev;            // device data
    dim3 gridDim, blockDim;
    blockDim.x = BLOCK_DIM;

    // determinazione esatta del numero di blocchi
    gridDim = SIZE / blockDim.x + ((SIZE % blockDim.x) == 0 ? 0 : 1);

    // Allocazione memoria
    int nBytes = SIZE * sizeof(float);

    a_host = (float *)malloc(nBytes);
    b_host = (float *)malloc(nBytes);
    res_host_gpu = (float *)malloc(nBytes);
    res_host_cpu = (float *)malloc(nBytes);

    cudaMalloc((void **)&a_dev, nBytes);
    cudaMalloc((void **)&b_dev, nBytes);
    cudaMalloc((void **)&c_dev, nBytes);

    // inizializzo i dati
    srand((unsigned int)time(0));

    for (int i = 0; i < SIZE; i++) {
        a_host[i] = rand() % 5 - 2;
        b_host[i] = rand() % 5 - 2;
    }

    cudaMemcpy(a_dev, a_host, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b_host, nBytes, cudaMemcpyHostToDevice);

    // azzeriamo il contenuto del vettore c
    memset(res_host_gpu, 0, nBytes);
    cudaMemset(c_dev, 0, nBytes);

    // invocazione del kernel
    GPU_SUM<<<gridDim, blockDim>>>(a_dev, b_dev, c_dev, SIZE);
    cudaMemcpy(res_host_gpu, c_dev, nBytes, cudaMemcpyDeviceToHost);

    // calcolo somma seriale su CPU
    CPU_SUM(a_host, b_host, res_host_cpu, SIZE);

    // verifica che i risultati di CPU e GPU siano uguali
    for (int i = 0; i < SIZE; i++) assert(res_host_gpu[i] == res_host_cpu[i]);

    if (SIZE < 20) {
        printf("Vettore A:");
        printVet(a_host, SIZE);
        printf("Vettore B:");
        printVet(b_host, SIZE);
        printf("Vettore result della GPU:");
        printVet(res_host_gpu, SIZE);
        printf("Vettore result della CPU:");
        printVet(res_host_cpu, SIZE);
    }

    free(a_host);
    free(b_host);
    free(res_host_gpu);
    free(res_host_cpu);

    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);

    return 0;
}