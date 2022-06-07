#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <chrono>
#include <iostream>

#define SIZE 8
#define MASK_SIZE 3
#define MASK_CENTER 1

void printMat(float* mat, int size) {
    // Print the entire matrix
    printf("\n");
    for (int i = 0; i < (size * size); i++) {
        printf("|");
        printf("%05.2f", mat[i]);
        if (((i + 1) % (size) == 0) && (i != 0))
            printf("|\n");
        if ((size * size) == 1)
            printf("|\n");
        if (size == 1 && ((i == 0)))
            printf("|\n");
    }
    printf("\n");
}

int main() {
    float* mat_start = (float*)malloc(SIZE * SIZE * sizeof(float));
    float* mat_res = (float*)calloc(SIZE * SIZE, sizeof(float));

    float* mask = (float*)malloc(MASK_SIZE * MASK_SIZE * sizeof(float));

    for (int i = 0; i < SIZE * SIZE; i++) {
        mat_start[i] = 1;
    }

    for (int i = 0; i < MASK_SIZE * MASK_SIZE; i++) {
        mask[i] = 1;
    }

    for (int mat_row = 0; mat_row < SIZE; mat_row++) {
        for (int mat_col = 0; mat_col < SIZE; mat_col++) {
            // printf("\n------------------------------------------\n\n");
            for (int k_row = 0; k_row < MASK_SIZE; k_row++) {
                for (int k_col = 0; k_col < MASK_SIZE; k_col++) {
                    int rel_row = mat_row + (k_row - MASK_CENTER);
                    int rel_col = mat_col + (k_col - MASK_CENTER);

                    // printf("Posizione Generale [%d,%d]; \n", mat_row, mat_col);
                    // printf("Posizione Mat[%d, %d], Mask[%d,%d]; \n", rel_row, rel_col, k_row, k_col);

                    if (rel_row >= 0 && rel_row < SIZE && rel_col >= 0 && rel_col < SIZE) {
                        // printf("Elementi moltiplicati input %05.2f, kernel %05.2f; \n\n", mat_start[(rel_row * SIZE) + rel_col], mask[(k_row * MASK_SIZE) + k_col]);
                        mat_res[(mat_row * SIZE) + mat_col] += mat_start[(rel_row * SIZE) + rel_col] * mask[(k_row * MASK_SIZE) + k_col];
                    } else {
                        // printf("\n");
                    }
                }
            }
        }
    }
    printf("\n------------------------------------------\n\n");
    printf("Matrice iniziale:");
    printMat(mat_start, SIZE);
    printf("Maschera:");
    printMat(mask, MASK_SIZE);
    printf("Matrice risultate:");
    printMat(mat_res, SIZE);

    free(mat_start);
    free(mat_res);

    free(mask);
}