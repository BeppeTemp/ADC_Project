#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <chrono>
#include <iostream>

#define MASK_SIZE 2
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
    int sizes[5] = {4, 4,4,4};

    float *mat_start, *mat_res;
    float* mask = (float*)malloc(MASK_SIZE * MASK_SIZE * sizeof(float));

    for (int i = 0; i < MASK_SIZE * MASK_SIZE; i++) {
        mask[i] = 1;
    }

    for (int k = 0; k < 5; k++) {
        mat_start = (float*)malloc(sizes[k] * sizes[k] * sizeof(float));
        mat_res = (float*)calloc(sizes[k] * sizes[k], sizeof(float));

        for (int i = 0; i < sizes[k] * sizes[k]; i++) {
            mat_start[i] = 1;
        }

        auto start = high_resolution_clock::now();

        #pragma omp parallel for 
        for (int mat_row = 0; mat_row < sizes[k]; mat_row++)
            #pragma omp parallel for
            for (int mat_col = 0; mat_col < sizes[k]; mat_col++)
                for (int k_row = 0; k_row < MASK_SIZE; k_row++)
                    for (int k_col = 0; k_col < MASK_SIZE; k_col++) {
                        int rel_row = mat_row + (k_row - MASK_CENTER);
                        int rel_col = mat_col + (k_col - MASK_CENTER);

                        if (rel_row >= 0 && rel_row < sizes[k] && rel_col >= 0 && rel_col < sizes[k]) {
                            mat_res[(mat_row * sizes[k]) + mat_col] += mat_start[(rel_row * sizes[k]) + rel_col] * mask[(k_row * MASK_SIZE) + k_col];
                        }
                    }

        auto stop = high_resolution_clock::now();

        printf("Matrix size: %d x %d \n", sizes[k], sizes[k]);
        time_stats(duration_cast<microseconds>(stop - start).count());

        free(mat_start);
        free(mat_res);
    }
    free(mask);
    return 0;
}