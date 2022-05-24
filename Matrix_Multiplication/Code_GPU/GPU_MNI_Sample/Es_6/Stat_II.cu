#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <time.h>

#define SIZE 4
#define BLOCK_DIM 64

// Somma Parallela
__global__ void GPU_SCALAR_PRODUCT(float *a, float *b, float *c, int step, int *vetp, int n) {
    extern __shared__ float v[];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int id = threadIdx.x;

    if (index < SIZE) {
        v[id] = a[index] * b[index];

        __syncthreads();

        for (int k = 0; k < step; k++) {
            if ((id % vetp[k + 1]) == 0) {
                v[id] += v[id + vetp[k]];
            }
            __syncthreads();
        }

        __syncthreads();

        if (id == 0) c[blockIdx.x] = v[0];
    }
}
// Stampa vettore
void printVet(float *vet, int size) {
    printf("\n");
    for (int i = 0; i < size; i++) {
        printf("|%05.2f", vet[i]);
    }
    printf("|\n\n");
}
// Calcolo del logaritmo in base due
int logBaseTwo(int value) {
    int result = 0;

    while (value != 1) {
        value = value >> 1;
        result++;
    }
    return result;
}

int main(void) {
    float *a_host, *b_host, *res_host_gpu;
    float *a_dev, *b_dev, *c_dev;
    int *vetp_host, *vetp_dev;
    dim3 gridDim, blockDim;

    // Configurazione del kernel
    blockDim.x = BLOCK_DIM;
    gridDim.x = SIZE / blockDim.x + ((SIZE % blockDim.x) == 0 ? 0 : 1);

    int step = logBaseTwo(blockDim.x);

    printf("Input: %d\n", SIZE);
    printf("Thread per Blocco: %d\n", blockDim.x);
    printf("Numero di blocchi: %d\n", gridDim.x);
    printf("step: %d\n", step);
    printf("\n");

    // Allocazione memoria
    int nBytes = SIZE * sizeof(float);
    a_host = (float *)malloc(nBytes);
    b_host = (float *)malloc(nBytes);
    res_host_gpu = (float *)calloc(gridDim.x, sizeof(float));
    vetp_host = (int *)malloc((step + 1) * sizeof(int));

    cudaMalloc((void **)&a_dev, nBytes);
    cudaMalloc((void **)&b_dev, nBytes);
    cudaMalloc((void **)&c_dev, gridDim.x * sizeof(float));
    cudaMalloc((void **)&vetp_dev, (step + 1) * sizeof(int));

    // Generazione del vettore potenza
    int value = 1;
    for (int i = 0; i <= step; i++) {
        vetp_host[i] = value << i;
    }

    // Inizializzo i dati
    srand((unsigned int)time(0));
    for (int i = 0; i < SIZE; i++) {
        a_host[i] = rand() % 5 - 2;
        b_host[i] = rand() % 5 - 2;
    }

    // Copia dei dati dall'host al device
    cudaMemcpy(a_dev, a_host, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b_host, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(vetp_dev, vetp_host, (step + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // Azzeriamo il contenuto del vettore c
    cudaMemset(c_dev, 0, gridDim.x * sizeof(float));

    // Creazione eventi
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    GPU_SCALAR_PRODUCT<<<gridDim, blockDim, blockDim.x * sizeof(float)>>>(a_dev, b_dev, c_dev, step, vetp_dev, SIZE);

    // Fine computazione su GPU
    cudaEventRecord(stop);

    // Attende la terminazione di tutti thread
    cudaEventSynchronize(stop);

    // Tempo tra i due eventi in millisecondi su GPU
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("Tempo GPU=%f\n", elapsed);

    // Copia dei dati dall'device al host
    cudaMemcpy(res_host_gpu, c_dev, gridDim.x * sizeof(float), cudaMemcpyDeviceToHost);

    // Calcolo valore finale
    float res = 0;
    for (int i = 0; i < gridDim.x; i++) {
        res += res_host_gpu[i];
    }

    printf("\n");
    if (SIZE < 20) {
        printf("Vettore A:");
        printVet(a_host, SIZE);
        printf("Vettore B:");
        printVet(b_host, SIZE);
        printf("Vettore result della GPU:");
        printVet(res_host_gpu, gridDim.x);
    }
    printf("Prodotto finale: %.2f\n", res);

    free(a_host);
    free(b_host);
    free(res_host_gpu);

    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);

    return 0;
}