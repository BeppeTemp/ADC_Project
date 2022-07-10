#include <cuda.h>
#include <omp.h>
#include <mma.h>
#include <iostream>

#define BLOCK_DIM 32
#define TILE_WIDTH 32

#define TILE_CPU 64

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

double mm_cpu(float* mat_a, float* mat_b, float* mat_res, int size);
double mm_gpu(float* mat_a, float* mat_b, float* mat_res, int size);
double mm_tensor(half* mat_a, half* mat_b, float* mat_res, int size);

bool mm_checker(float* mat_res, int size);