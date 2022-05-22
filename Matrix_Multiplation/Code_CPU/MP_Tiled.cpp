#include <omp.h>
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono;

#define TILE 32

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

    for (int k = 0; k < 5; k++) {
        long nBytes = sizes[k] * sizes[k] * sizeof(float);

        mat_a = (float*)malloc(nBytes);
        mat_b = (float*)malloc(nBytes);
        mat_res = (float*)malloc(nBytes);

        for (int j = 0; j < sizes[k] * sizes[k]; j++) {
            mat_a[j] = 1;
            mat_b[j] = 1;
        }

        auto start = high_resolution_clock::now();

        #pragma omp parallel for
        for (int ih = 0; ih < sizes[k]; ih += TILE)
            #pragma omp parallel for
            for (int jh = 0; jh < sizes[k]; jh += TILE)
                for (int kh = 0; kh < sizes[k]; kh += TILE)
                    for (int il = 0; il < TILE; il++)
                        for (int kl = 0; kl < TILE; kl++)
                            for (int jl = 0; jl < TILE; jl++)
                                mat_res[(ih + il) * TILE + (jh + jl)] += mat_a[(ih + il) * TILE + (kh + kl)] * mat_b[(kh + kl) * TILE + (jh + jl)];

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