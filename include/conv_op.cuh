#include <cuda.h>
#include <omp.h>
#include <iostream>

#define MASK_SIZE 5
#define MASK_CENTER 2

#define BLOCK_DIM 32
#define TILE_WIDTH 32


double conv_cpu(float* mat_start, float* mask, float* mat_res, int mat_size);
double conv_gpu(float* mat_start, float* mask, float* mat_res, int mat_size);

bool conv_checker(float* mat_a, float* mat_b, int size);