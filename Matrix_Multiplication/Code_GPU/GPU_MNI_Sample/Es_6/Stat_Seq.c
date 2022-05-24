#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 4

// Stampa vettore
void printVet(float *vet, int size) {
    printf("\n");
    for (int i = 0; i < size; i++) {
        printf("|%05.2f", vet[i]);
    }
    printf("|\n\n");
}

int main(void) {
    float res = 0;
    float *a_host, *b_host, *c_host;
    double time_spent = 0.0;

    // Allocazione memoria
    int nBytes = SIZE * sizeof(float);
    a_host = malloc(nBytes);
    b_host = malloc(nBytes);
    c_host = malloc(nBytes);

    // Inizializzo i dati
    srand((unsigned int)time(0));
    for (int i = 0; i < SIZE; i++) {
        a_host[i] = rand() % 5 - 2;
        b_host[i] = rand() % 5 - 2;
    }

    clock_t begin = clock();

    for (int i = 0; i < SIZE; i++) {
        c_host[i] = a_host[i] * b_host[i];
        res += c_host[i];
    }

    clock_t end = clock();

    time_spent = ((double)(end - begin) / CLOCKS_PER_SEC) * 1000;

    printf("\n");
    if (SIZE < 20) {
        printf("Vettore A:");
        printVet(a_host, SIZE);
        printf("Vettore B:");
        printVet(b_host, SIZE);
        printf("Vettore parziale:");
        printVet(c_host, SIZE);
    }
    printf("Tempo CPU=%f\n\n", time_spent);
    printf("Prodotto finale: %.2f\n", res);

    free(a_host);
    free(b_host);
    free(c_host);

    return 0;
}