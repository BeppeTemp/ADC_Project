#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <chrono>
#include <iostream>

#define ARRAY_SIZE 8
#define MASK_SIZE 3

#define T_ARRAY_M (ARRAY_SIZE - MASK_SIZE + 1) * (ARRAY_SIZE - MASK_SIZE + 1)
#define T_ARRAY_N MASK_SIZE* MASK_SIZE

#define MASK_CENTER 1

using namespace std;

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
    float* mask = (float*)malloc(MASK_SIZE * MASK_SIZE * sizeof(float));

    for (int i = 0; i < MASK_SIZE; i++) {
        mask[i] = 1;
    }

    float* mat_start = (float*)calloc(ARRAY_SIZE * ARRAY_SIZE, sizeof(float));

    for (int i = 1; i < ARRAY_SIZE - 1; i++)
        for (int j = 1; j < ARRAY_SIZE - 1; j++)
            mat_start[i * ARRAY_SIZE + j] = 1;

    printMat(mat_start, ARRAY_SIZE, ARRAY_SIZE);

    float* mat_start_t = (float*)calloc(T_ARRAY_M * T_ARRAY_N, sizeof(float));

    int v = 0;

    int var_x = (ARRAY_SIZE / 2) + 1;
    int var_y = (ARRAY_SIZE / 2) + 1;

    printf("---------- \n");
    for (int t = 0; t < ARRAY_SIZE - MASK_SIZE + 1; t++) {
        for (int h = 0; h < ARRAY_SIZE - MASK_SIZE + 1; h++) {
            // printf("Stampo da [%d, %d] a [%d,%d]\n", t, h, ARRAY_SIZE - var_x - 1, ARRAY_SIZE - var_y - 1);
            for (int i = t; i < ARRAY_SIZE - var_x; i++) {
                for (int j = h; j < ARRAY_SIZE - var_y; j++) {
                    // printf("[%d,%d] ", i, j);
                    printf("%01.0f ", mat_start[i * ARRAY_SIZE + j]);
                    mat_start_t[v] = mat_start[i * ARRAY_SIZE + j];
                    v++;
                }
            }
            var_y--;
            printf("\n");
        }
        var_y = (ARRAY_SIZE / 2) + 1;
        var_x--;
        printf("---------- \n");
    }

    printMat(mat_start_t, T_ARRAY_M, T_ARRAY_N);

    float* mat_res = (float*)malloc((ARRAY_SIZE - 2) * (ARRAY_SIZE - 2) * sizeof(float));

    for (int i = 0; i < T_ARRAY_M; i++)
        for (int j = 0; j < 1; j++)
            for (int k = 0; k < MASK_SIZE * MASK_SIZE; k++)
                mat_res[i * (ARRAY_SIZE - 2) + j] += mat_start_t[i * T_ARRAY_N + k] * mask[k * MASK_SIZE + j];

    printMat(mat_res, ARRAY_SIZE - 2, ARRAY_SIZE - 2);

    free(mat_start_t);
    return 0;
}