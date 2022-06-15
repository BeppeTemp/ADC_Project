#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#define ARRAY_SIZE 4
#define MASK_SIZE 5

#define PADDED_ARRAY_SIZE (ARRAY_SIZE + ((MASK_SIZE / 2) * 2))

#define UNF_ARRAY_M ((PADDED_ARRAY_SIZE - MASK_SIZE + 1) * (PADDED_ARRAY_SIZE - MASK_SIZE + 1))
#define UNF_ARRAY_N (MASK_SIZE * MASK_SIZE)

#define STEP_X (PADDED_ARRAY_SIZE - MASK_SIZE)
#define STEP_Y (PADDED_ARRAY_SIZE - MASK_SIZE)

// Unfolded Matrix o Variante Toeplix Matrix

using namespace std;

void printMat(float* mat, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        printf("|");
        printf(" %02.0f ", mat[i]);
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
    for (int t = 0; t < PADDED_ARRAY_SIZE - MASK_SIZE + 1; t++) {
        for (int h = 0; h < PADDED_ARRAY_SIZE - MASK_SIZE + 1; h++) {
            // printf("Stampo da [%d, %d] a [%d,%d]\n", t, h, PADDED_ARRAY_SIZE - step_x - 1, PADDED_ARRAY_SIZE - step_y - 1);
            for (int i = t; i < PADDED_ARRAY_SIZE - step_x; i++) {
                for (int j = h; j < PADDED_ARRAY_SIZE - step_y; j++) {
                    // printf("[%d,%d] ", i, j);
                    // printf("%01.0f ", mat_start[i * PADDED_ARRAY_SIZE + j]);
                    mat_unfolded[k] = mat_start[i * PADDED_ARRAY_SIZE + j];
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
    printf("Size: %d \n", ARRAY_SIZE);
    printf("Mask: %d \n", MASK_SIZE);
    printf("Mask/2: %d \n", MASK_SIZE / 2);
    printf("Padded: %d \n", PADDED_ARRAY_SIZE);
    printf("Unf_Array_M: %d \n", UNF_ARRAY_M);
    printf("Unf_Array_N: %d \n", UNF_ARRAY_N);
    printf("Step_X: %d \n", STEP_X);
    printf("Step_Y: %d \n\n", STEP_Y);

    float* mask = (float*)calloc(UNF_ARRAY_N * UNF_ARRAY_M, sizeof(float));

    for (int i = 0; i < UNF_ARRAY_N; i++)
            mask[i * UNF_ARRAY_M] = 1;

    printf("Maschera in column major con padding:\n");
    printMat(mask, UNF_ARRAY_N, UNF_ARRAY_M);

    float* mat_start = (float*)calloc(PADDED_ARRAY_SIZE * PADDED_ARRAY_SIZE, sizeof(float));

    for (int i = MASK_SIZE / 2; i < PADDED_ARRAY_SIZE - (MASK_SIZE / 2); i++) {
        for (int j = MASK_SIZE / 2; j < PADDED_ARRAY_SIZE - (MASK_SIZE / 2); j++) {
            mat_start[i * PADDED_ARRAY_SIZE + j] = 1;
        }
    }

    printf("Matrice iniziale con padding:\n");
    printMat(mat_start, PADDED_ARRAY_SIZE, PADDED_ARRAY_SIZE);

    float* mat_unfolded = (float*)calloc(UNF_ARRAY_M * UNF_ARRAY_N, sizeof(float));
    matrixUnfold(mat_start, mat_unfolded);

    printf("Matrice finale unfolded:\n");
    printMat(mat_unfolded, UNF_ARRAY_M, UNF_ARRAY_N);

    float* mat_res_col = (float*)malloc(UNF_ARRAY_M * sizeof(float));

    for (int i = 0; i < UNF_ARRAY_M; i++) // Righe Unfolded
        for (int j = 0; j < UNF_ARRAY_M; j++) // Colonne Mask
            for (int k = 0; k < UNF_ARRAY_N; k++){ // Righe Mask
                mat_res_col[i] += mat_unfolded[i * UNF_ARRAY_N + k] * mask[k * UNF_ARRAY_M + j];
            }

    printf("Risultato finale della convolution:\n");
    printMat(mat_res_col, ARRAY_SIZE, ARRAY_SIZE);

    free(mask);
    free(mat_start);
    free(mat_unfolded);
    free(mat_res_col);

    return 0;
}