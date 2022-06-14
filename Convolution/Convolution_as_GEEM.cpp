#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <chrono>
#include <iostream>

#define ARRAY_SIZE 16
#define MASK_SIZE 5

#define PADDED_ARRAY_SIZE (ARRAY_SIZE + (MASK_SIZE / 2))

#define UNF_ARRAY_M (ARRAY_SIZE - MASK_SIZE + 1) * (ARRAY_SIZE - MASK_SIZE + 1)
#define UNF_ARRAY_N (MASK_SIZE* MASK_SIZE)

#define STEP_X (ARRAY_SIZE - MASK_SIZE)
#define STEP_Y (ARRAY_SIZE - MASK_SIZE)

using namespace std;

void printMat(float* mat, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        printf("|");
        printf(" %01.0f ", mat[i]);
        if (((i + 1) % (n) == 0) && (i != 0))
            printf("|\n");
        if ((m * n) == 1)
            printf("|\n");
        if (n == 1 && ((i == 0)))
            printf("|\n");
    }
    printf("\n");
}

void matrixUnfold(float* mat_start, float* mat_unfolded) {
    int step_x = STEP_X;
    int step_y = STEP_Y;

    int k = 0;
    for (int t = 0; t < ARRAY_SIZE - MASK_SIZE + 1; t++) {
        for (int h = 0; h < ARRAY_SIZE - MASK_SIZE + 1; h++) {
            // printf("Stampo da [%d, %d] a [%d,%d]\n", t, h, ARRAY_SIZE - step_x - 1, ARRAY_SIZE - step_y - 1);
            for (int i = t; i < ARRAY_SIZE - step_x; i++) {
                for (int j = h; j < ARRAY_SIZE - step_y; j++) {
                    // printf("[%d,%d] ", i, j);
                    // printf("%01.0f ", mat_start[i * ARRAY_SIZE + j]);
                    mat_unfolded[k] = mat_start[i * ARRAY_SIZE + j];
                    k++;
                }
            }
            step_y--;
            // printf("\n");
        }
        step_y = STEP_X;
        step_x--;
    }
}

int main() {
    float* mask = (float*)calloc(UNF_ARRAY_M * UNF_ARRAY_N, sizeof(float));

    int k = UNF_ARRAY_M;
    for (int i = 0; i < UNF_ARRAY_N; i++)
        if (k > 0) {
            mask[i * UNF_ARRAY_M] = 1;
            k--;
        }

    // printf("Maschera in column major con padding:\n");
    // printMat(mask, UNF_ARRAY_N, MASK_SIZE * MASK_SIZE);

    float* mat_start = (float*)calloc(PADDED_ARRAY_SIZE * PADDED_ARRAY_SIZE, sizeof(float));

    for (int i = MASK_SIZE / 2; i < ARRAY_SIZE; i++) {
        for (int j = MASK_SIZE / 2; j < ARRAY_SIZE; j++) {
            mat_start[i * PADDED_ARRAY_SIZE + j] = 1;
        }
    }

    printf("Matrice iniziale con padding:\n");
    printMat(mat_start, PADDED_ARRAY_SIZE, PADDED_ARRAY_SIZE);

    float* mat_unfolded = (float*)calloc(UNF_ARRAY_N * UNF_ARRAY_M, sizeof(float));
    matrixUnfold(mat_start, mat_unfolded);

    // printf("Matrice finale unfolded:\n");

    float* mat_res_col = (float*)malloc(UNF_ARRAY_N * sizeof(float));

    for (int i = 0; i < UNF_ARRAY_N; i++)
        for (int j = 0; j < UNF_ARRAY_M; j++)
            for (int k = 0; k < UNF_ARRAY_N; k++)
                mat_res_col[i] += mat_unfolded[i * UNF_ARRAY_M + k] * mask[k * UNF_ARRAY_M + j];

    printf("Risultato finale della convolution:\n");
    printMat(mat_res_col, 16, 16);

    return 0;
}