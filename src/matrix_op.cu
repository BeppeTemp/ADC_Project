#include <omp.h>
#include <cuda.h>
#include <mma.h>
#include <iostream>

using namespace nvcuda;

double mm_cpu(float* mat_res, int size) {
    double t_init = omp_get_wtime();

    float* mat_a = (float*)malloc(size * size * sizeof(float));
    float* mat_b = (float*)malloc(size * size * sizeof(float));

    for (int i = 0; i < size * size; i++) {
        mat_a[i] = (float)i;
        mat_b[i] = (float)i;
    }

    #pragma omp parallel for collapse(3)
    for (int mat_row = 0; mat_row < size; mat_row++)
        for (int mat_col = 0; mat_col < size; mat_col++)
            for (int k_row = 0; k_row < size; k_row++)
                mat_res[mat_row * size + mat_col] += mat_a[mat_row * size + k_row] * mat_b[k_row * size + mat_col];

    return omp_get_wtime() - t_init;
}

__global__ void MatrixMulKernelTiled(float* mat_a, float* mat_b, float* res_mat, int size) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int mat_a_begin = size * BLOCK_DIM * by;
    int mat_a_end = mat_a_begin + size - 1;
    int mat_b_begin = BLOCK_DIM * bx;

    int mat_b_step = BLOCK_DIM * size;

    float temp = 0;
    for (int a = mat_a_begin, b = mat_b_begin; a <= mat_a_end; a += BLOCK_DIM, b += mat_b_step) {
        __shared__ float m_a_sh[BLOCK_DIM][BLOCK_DIM];
        __shared__ float m_b_sh[BLOCK_DIM][BLOCK_DIM];

        m_a_sh[ty][tx] = mat_a[a + size * ty + tx];
        m_b_sh[ty][tx] = mat_b[b + size * ty + tx];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_DIM; ++k) {
            temp += m_a_sh[ty][k] * m_b_sh[k][tx];
        }

        __syncthreads();
    }

    int c = size * BLOCK_DIM * by + BLOCK_DIM * bx;
    res_mat[c + size * ty + tx] = temp;
}

double mm_gpu(float* mat_res, int size){

}