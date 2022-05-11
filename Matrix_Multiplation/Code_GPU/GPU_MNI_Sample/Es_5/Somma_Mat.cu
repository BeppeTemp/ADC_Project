#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <time.h>

#define SIZE 8192

#define BLOCK_DIM 5

// Somma Seriale
__host__ void CPU_SUM(float *mat_a, float *mat_b, float *res_mat) {
    int i;
    for (i = 0; i < SIZE * SIZE; i++) res_mat[i] = mat_a[i] + mat_b[i];
}
// Somma Parallela
__global__ void GPU_SUM(float *mat_a, float *mat_b, float *res_mat) {
    int row = threadIdx.x + (blockIdx.x * blockDim.x);
    int column = threadIdx.y + (blockIdx.y * blockDim.y);

    int index = (row * SIZE ) + column;

    if (row < SIZE && column < SIZE) res_mat[index] = mat_a[index] + mat_b[index];
}
// Stampa della matrice
void printMat(float *mat, int row, int column) {
    printf("\n");
    for (int i = 0; i < (row * column); i++) {
        printf("|");
        printf("%07.2f", mat[i]);
        if (((i + 1) % (column) == 0) && (i != 0)) printf("|\n");
        if ((row * column) == 1) printf("|\n");
        if (column == 1 && ((i == 0))) printf("|\n");
    }
    printf("\n");
}

int main(void) {
    float *mat_a_host, *mat_b_host, *mat_res_host_gpu, *mat_res_host_cpu;  // host data
    float *mat_a_dev, *mat_b_dev, *mat_res_dev;                            // device data
    dim3 gridDim, blockDim;

    // Allocazione memoria
    int nBytes = SIZE * SIZE * sizeof(float);

    mat_a_host = (float *)malloc(nBytes);
    mat_b_host = (float *)malloc(nBytes);
    mat_res_host_gpu = (float *)calloc(nBytes, sizeof(float));
    mat_res_host_cpu = (float *)calloc(nBytes, sizeof(float));

    cudaMalloc((void **)&mat_a_dev, nBytes);
    cudaMalloc((void **)&mat_b_dev, nBytes);
    cudaMalloc((void **)&mat_res_dev, nBytes);

    // Inizializzo i dati
    srand(time(NULL));
    for (int i = 0; i < SIZE * SIZE; i++) {
        mat_a_host[i] = i;
        mat_b_host[i] = i;
        // mat_a_host[i] = (double)rand() / RAND_MAX * 2.0;
        // mat_b_host[i] = (double)rand() / RAND_MAX * 2.0;
    }

    // Copia dei dati dall'host al device
    // cudaMemcpy(mat_a_dev, mat_a_host, nBytes, cudaMemcpyHostToDevice);
    // cudaMemcpy(mat_b_dev, mat_b_host, nBytes, cudaMemcpyHostToDevice);
    
    cudaMemcpy(mat_a_dev, mat_a_host, nBytes, cudaMemcpyDefault);
    cudaMemcpy(mat_b_dev, mat_b_host, nBytes, cudaMemcpyDefault);

    // Azzeriamo il contenuto del vettore c
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

    // Invocazione del kernel
    GPU_SUM<<<gridDim, blockDim>>>(mat_a_dev, mat_b_dev, mat_res_dev);

    // Fine computazione su GPU
    cudaEventRecord(stop);

    // Attende la terminazione di tutti thread
    cudaEventSynchronize(stop);

    // Tempo tra i due eventi in millisecondi su GPU
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("Tempo GPU=%f\n", elapsed);

    // Copia dei dati dall'device al host
    cudaMemcpy(mat_res_host_gpu, mat_res_dev, nBytes, cudaMemcpyDeviceToHost);

    // Inizio computazione su CPU
    cudaEventRecord(start);

    // calcolo somma seriale
    CPU_SUM(mat_a_host, mat_b_host, mat_res_host_cpu);

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
    // for (int i = 0; i < ROW * COLUMN; i++) assert(mat_res_host_gpu[i] == mat_res_host_cpu[i]);

    printf("\n");
    if (SIZE <= 32) {
        printf("Matrice A:");
        printMat(mat_a_host, SIZE, SIZE);
        printf("Matrice B:");
        printMat(mat_b_host, SIZE, SIZE);
        printf("Vettore result della GPU:");
        printMat(mat_res_host_gpu, SIZE, SIZE);
        printf("Vettore result della CPU:");
        printMat(mat_res_host_cpu, SIZE, SIZE);
    }

    free(mat_a_host);
    free(mat_b_host);
    free(mat_res_host_gpu);
    free(mat_res_host_cpu);

    cudaFree(mat_a_dev);
    cudaFree(mat_b_dev);
    cudaFree(mat_res_dev);

    return 0;
}