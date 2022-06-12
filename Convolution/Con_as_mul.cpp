#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <chrono>
#include <iostream>

#define ARRAY_SIZE 3
#define MASK_SIZE 2

#define T_MASK_M (ARRAY_SIZE - MASK_SIZE + 1) * (ARRAY_SIZE - MASK_SIZE + 1)
#define T_MASK_N ARRAY_SIZE* ARRAY_SIZE

#define MASK_CENTER 1

using namespace std;
using namespace std::chrono;

void time_stats(float micro_seconds) {
    printf("Execution times:\n");
    printf("    * %.0f μs \n", micro_seconds);
    printf("    * %.2f ms \n", micro_seconds / 1000);
    printf("    * %.2f s \n", micro_seconds / 1000 / 1000);
    printf("\n");
}

void printMat(float* mat, int m, int n) {
    // Print the entire matrix
    printf("\n");
    for (int i = 0; i < m * n; i++) {
        printf("|");
        printf("%03.1f", mat[i]);
        if (((i + 1) % (n) == 0) && (i != 0))
            printf("|\n");
        if ((m * n) == 1)
            printf("|\n");
        if (n == 1 && ((i == 0)))
            printf("|\n");
    }

    printf("\n");
}

int main() {
    float *mat_start, *mat_res;
    float* mask = (float*)calloc(T_MASK_M * T_MASK_N, sizeof(float));

    printf("Row: %d\n", T_MASK_M);
    printf("Col: %d\n", T_MASK_N);

    int x = 0;

    for (int i = 0; i < T_MASK_M; i++) {
        for (int j = i; j < T_MASK_N; j++) {
            if (j & MASK_SIZE != 0) {
                mask[(i * T_MASK_N) + x] = 1;
                x++;
            }
        }
        x = i;
    }

    printMat(mask, T_MASK_M, T_MASK_N);

    free(mask);
    return 0;
}