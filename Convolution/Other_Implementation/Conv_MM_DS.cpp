#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#define ARRAY_SIZE 6
#define MASK_SIZE 5

#define PADDED_ARRAY_SIZE (ARRAY_SIZE + ((MASK_SIZE / 2) * 2))

#define STEP_X (PADDED_ARRAY_SIZE - MASK_SIZE)
#define STEP_Y (PADDED_ARRAY_SIZE - MASK_SIZE)

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
int diagonalSum(float* mat, int m, int n) {
    // TODO Questo può essere sicuramente ottimizzato

    int sum = 0;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (i == j)
                sum += mat[(i * n) + j];
    return sum;
}

int main() {
    printf("Size: %d \n", ARRAY_SIZE);
    printf("Mask: %d \n", MASK_SIZE);
    printf("Mask/2: %d \n", MASK_SIZE / 2);
    printf("Padded: %d \n", PADDED_ARRAY_SIZE);
    printf("\n");

    float* mask = (float*)calloc(MASK_SIZE * MASK_SIZE, sizeof(float));

    for (int i = 0; i < MASK_SIZE * MASK_SIZE; i++)
        mask[i] = 1;
        // mask[i] = i;

    printf("Transposed mask:\n");
    printMat(mask, MASK_SIZE, MASK_SIZE);

    float* mat_start = (float*)calloc(PADDED_ARRAY_SIZE * PADDED_ARRAY_SIZE, sizeof(float));

    for (int i = MASK_SIZE / 2; i < PADDED_ARRAY_SIZE - (MASK_SIZE / 2); i++) {
        for (int j = MASK_SIZE / 2; j < PADDED_ARRAY_SIZE - (MASK_SIZE / 2); j++) {
            mat_start[i * PADDED_ARRAY_SIZE + j] = 1;
        }
    }

    // for (int i = 0; i < PADDED_ARRAY_SIZE * PADDED_ARRAY_SIZE; i++) {
    //     mat_start[i] = i;
    // }

    printf("Initial matrix padded:\n");
    printMat(mat_start, PADDED_ARRAY_SIZE, PADDED_ARRAY_SIZE);

    float* mat_res = (float*)calloc(ARRAY_SIZE * ARRAY_SIZE, sizeof(float));

    int step_x = STEP_X;
    int step_y = STEP_Y;

    for (int t = 0; t < PADDED_ARRAY_SIZE - MASK_SIZE + 1; t++) {
        for (int h = 0; h < PADDED_ARRAY_SIZE - MASK_SIZE + 1; h++) {
            float* mat_res_temp = (float*)calloc(MASK_SIZE * MASK_SIZE, sizeof(float));

            int res_x = t + (MASK_SIZE / 2) - (MASK_SIZE / 2);
            int res_y = h + (MASK_SIZE / 2) - (MASK_SIZE / 2);

            // printf("Moltiplico da [%d, %d] a [%d,%d] e inserisco in [%d,%d] \n", t, h, PADDED_ARRAY_SIZE - step_x - 1, PADDED_ARRAY_SIZE - step_y - 1, res_x, res_y);

            for (int i = t; i < PADDED_ARRAY_SIZE - step_x; i++) {  // row mat start
                for (int j = 0; j < MASK_SIZE; j++) {
                    int mi = 0;
                    for (int k = h; k < PADDED_ARRAY_SIZE - step_y; k++)  // col mat start
                    {
                        // printf("[%d,%d][%d,%d] ", i, k, mi, j);
                        // printf("[%05.2f][%05.2f] ", mat_start[i * PADDED_ARRAY_SIZE + k], mask[mi * MASK_SIZE + j]);
                        mat_res_temp[mi * MASK_SIZE + j] += mat_start[i * PADDED_ARRAY_SIZE + k] * mask[mi * MASK_SIZE + j];
                        mi++;
                    }
                    // printf("\n");
                }
                // TODO è un problema di j
            }
            // printf("la somma della diagonale è %d\n\n", diagonalSum(mat_res_temp, MASK_SIZE, MASK_SIZE));
            // printMat(mat_res_temp, MASK_SIZE, MASK_SIZE);

            mat_res[(res_x * ARRAY_SIZE) + res_y] = diagonalSum(mat_res_temp, MASK_SIZE, MASK_SIZE);

            free(mat_res_temp);

            step_y--;
        }
        step_y = STEP_X;
        step_x--;
    }

    printf("Matrice finale:\n");
    printMat(mat_res, ARRAY_SIZE, ARRAY_SIZE);

    return 0;
}