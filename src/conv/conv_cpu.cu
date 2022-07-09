void conv_cpu(float* mat_start, float* mask, float* mat_res, int mat_size, int mask_size, int mask_center) {
    #pragma omp parallel for
    for (int mat_row = 0; mat_row < mat_size; mat_row++)
    #pragma omp parallel for
        for (int mat_col = 0; mat_col < mat_size; mat_col++)
            for (int k_row = 0; k_row < mask_size; k_row++)
                for (int k_col = 0; k_col < mask_size; k_col++) {
                    int rel_row = mat_row + (k_row - mask_center);
                    int rel_col = mat_col + (k_col - mask_center);

                    if (rel_row >= 0 && rel_row < mat_size && rel_col >= 0 && rel_col < mat_size) {
                        mat_res[(mat_row * mat_size) + mat_col] += mat_start[(rel_row * mat_size) + rel_col] * mask[(k_row * mask_size) + k_col];
                    }
                }
}