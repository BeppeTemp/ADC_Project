#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

void time_stats(float micro_seconds) {
    printf("Execution times:\n");
    printf("    * %.0f μs \n", micro_seconds);
    printf("    * %.2f ms \n", micro_seconds / 1000);
    printf("    * %.2f s \n", micro_seconds / 1000 / 1000);
    printf("\n");
}

int main() {
    int sizes[5] = {1024, 2048, 4096, 8192, 16384};

    float *mat_a, *mat_b, *mat_res;

    srand(117);

    for (int k = 0; k < 5; k++) {
        long nBytes = sizes[k] * sizes[k] * sizeof(float);

        mat_a = (float*)malloc(nBytes);
        mat_b = (float*)malloc(nBytes);
        mat_res = (float*)malloc(nBytes);

        for (int j = 0; j < sizes[k] * sizes[k]; j++) {
            mat_a[j] = (float)(rand() / (float)(RAND_MAX));
            mat_b[j] = (float)(rand() / (float)(RAND_MAX));
        }

        auto start = high_resolution_clock::now();
        for (int i = 0; i < sizes[k]; i++)
            for (int j = 0; j < sizes[k]; j++)
                for (int l = 0; l < sizes[k]; l++) {
                    mat_res[i * sizes[k] + j] += mat_a[i * sizes[k] + l] * mat_b[l * sizes[k] + j];
                }
        auto stop = high_resolution_clock::now();

        long check = 0;
        for (int t = 0; t < sizes[k] * sizes[k]; t++) {
            check += (long)mat_res[t];
        }

        printf("Matrix size: %d x %d \n", sizes[k], sizes[k]);
        printf("Check: %ld\n", check);
        time_stats(duration_cast<microseconds>(stop - start).count());

        free(mat_a);
        free(mat_b);
        free(mat_res);
    }

    return 0;
}