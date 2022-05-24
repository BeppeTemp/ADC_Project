#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <time.h>

#define SIZE 4
#define BLOCK_DIM 64

// Somma Seriale
__host__ void CPU_SUM(float *a, float *b, float *c, int n) {
    int i;
    for (i = 0; i < n; i++) c[i] = a[i] + b[i];
}
// Somma Parallela
__global__ void GPU_SUM(float *a, float *b, float *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
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
    float *a_dev, *b_dev, *c_dev;                          // device data
    dim3 gridDim, blockDim;

    // Allocazione memoria
    int nBytes = SIZE * sizeof(float);

    a_host = (float *)malloc(nBytes);
    b_host = (float *)malloc(nBytes);
    res_host_gpu = (float *) calloc(nBytes, sizeof(float));
    res_host_cpu = (float *) calloc(nBytes, sizeof(float));;

    cudaMalloc((void **)&a_dev, nBytes);
    cudaMalloc((void **)&b_dev, nBytes);
    cudaMalloc((void **)&c_dev, nBytes);

    // Inizializzo i dati
    srand((unsigned int)time(0));

    for (int i = 0; i < SIZE; i++) {
        a_host[i] = rand() % 5 - 2;
        b_host[i] = rand() % 5 - 2;
    }

	// Copia dei dati dall'host al device
    cudaMemcpy(a_dev, a_host, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b_host, nBytes, cudaMemcpyHostToDevice);

    // Azzeriamo il contenuto del vettore c
    cudaMemset(c_dev, 0, nBytes);

    // Configurazione del kernel
    blockDim.x = BLOCK_DIM;
    gridDim.x = SIZE / blockDim.x + ((SIZE % blockDim.x) == 0 ? 0 : 1);

	// Creazione eventi
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	// Inizio computazione su GPU
    cudaEventRecord(start);

    // Invocazione del kernel
    GPU_SUM<<<gridDim, blockDim>>>(a_dev, b_dev, c_dev, SIZE);

	// Fine computazione su GPU
    cudaEventRecord(stop);

	// Attende la terminazione di tutti thread
    cudaEventSynchronize(stop);

	// Tempo tra i due eventi in millisecondi su GPU
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("Tempo GPU=%f\n", elapsed);

	// Copia dei dati dall'device al host
    cudaMemcpy(res_host_gpu, c_dev, nBytes, cudaMemcpyDeviceToHost);

    // Inizio computazione su CPU
    cudaEventRecord(start);

    // calcolo somma seriale
    CPU_SUM(a_host, b_host, res_host_cpu, SIZE);

	// Fine computazione su CPU
    cudaEventRecord(stop);

	// Attende la terminazione di tutti thread
    cudaEventSynchronize(stop);

	// Tempo tra i due eventi in millisecondi su GPU
    cudaEventElapsedTime(&elapsed, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("Tempo CPU=%f\n", elapsed);

    // verifica che i risultati di CPU e GPU siano uguali
    for (int i = 0; i < SIZE; i++) assert(res_host_gpu[i] == res_host_cpu[i]);

    printf("\n");
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