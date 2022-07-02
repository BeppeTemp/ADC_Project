#include <iostream>
#include <boost/program_options.hpp>

#include "../include/conv_cpu.hpp"
#include "../include/mm_cpu.hpp"

using namespace std;

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
    int size;

    cout << "Inserire grandezza matrice: ";
    cin >> size;

    float *mat_a, *mat_b, *mat_res;

    mat_a = (float*)malloc(size * size * sizeof(float));
    mat_b = (float*)malloc(size * size * sizeof(float));
    mat_res = (float*)calloc(size * size, sizeof(float));

    for (int i = 0; i < size * size; i++) {
        mat_a[i] = 1;
        mat_b[i] = 1;
    }

    mm_cpu(mat_a, mat_b, mat_res, size);
    cout << "Risultato matrice a * b: ";
    printMat(mat_res, size);

    int mask_size;

    cout << "Inserire grandezza maschera: ";
    cin >> mask_size;

    float *mask, *mat_start;

    mat_start = (float*)malloc(size * size * sizeof(float));
    mask = (float*)malloc(mask_size * mask_size * sizeof(float));

    conv_cpu(mat_start, mask, mat_res, size, mask_size, mask_size / 2);
    cout << "Risultato conv: ";
    printMat(mat_res, size);

    return 0;
}