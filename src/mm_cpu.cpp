void mm_cpu(float* mat_a, float* mat_b, float* mat_res, int size) {
#pragma omp parallel for
    for (int mat_row = 0; mat_row < size; mat_row++)
        for (int mat_col = 0; mat_col < size; mat_col++)
            for (int k_row = 0; k_row < size; k_row++)
                mat_res[mat_row * size + mat_col] += mat_a[mat_row * size + k_row] * mat_b[k_row * size + mat_col];
}