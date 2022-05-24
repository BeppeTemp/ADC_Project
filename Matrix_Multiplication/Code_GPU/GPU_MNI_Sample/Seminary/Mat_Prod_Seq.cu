#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <time.h>

#define SIZE 4

// Stampa della matrice
void printMat(float* mat, int row, int column) {
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
    float *a_host, *b_host, *c_host;

    // Allocazione memoria
    int nBytes = SIZE * SIZE * sizeof(float);
    a_host = (float *) malloc(nBytes);
    b_host = (float *) malloc(nBytes);
    c_host = (float *) malloc(nBytes);

    // Inizializzo i dati
    srand(117);
    for (int i = 0; i < SIZE * SIZE; i++) {
        a_host[i] = i;
        b_host[i] = i;
        // a_host[i] = rand() % 5 - 2;
        // b_host[i] = rand() % 5 - 2;
    }

        // Creazione eventi
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++)
            for (int k = 0; k < SIZE; k++) {
                c_host[(i * SIZE) + j] += a_host[(i * SIZE) + k] * b_host[(k * SIZE) + j];
            }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("Tempo CPU=%f\n\n", elapsed);

    printf("\n");
    if (SIZE <= 8) {
        printf("Matrice A:");
        printMat(a_host, SIZE, SIZE);
        printf("Matrice B:");
        printMat(b_host, SIZE, SIZE);
        printf("Matrice result della CPU:");
        printMat(c_host, SIZE, SIZE);
    }

    free(a_host);
    free(b_host);
    free(c_host);

    return 0;
}