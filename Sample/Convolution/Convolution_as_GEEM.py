from scipy import signal
from scipy import misc
import numpy as np
from numpy import zeros


def unfold_matrix(X, mask_size):
    mat_row, mat_col = X.shape[0:2]

    xx = zeros(((mat_row - mask_size + 1) * (mat_col - mask_size + 1), mask_size**2))

    row_num = 0

    def make_row(mat):
        return mat.flatten()

    for i in range(mat_row - mask_size + 1):
        for j in range(mat_col - mask_size + 1):
            # collect block of mat_col*mat_col elements and convert to row
            xx[row_num, :] = make_row(X[i : i + mask_size, j : j + mask_size])
            row_num = row_num + 1

    return xx


kernel = np.array(
    [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1]],
    np.float32,
)

mat = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
)

mat_row, mat_col = mat.shape[0:2]

mask_size = kernel.shape[0]

x_unfolded = unfold_matrix(mat, mask_size)

print(x_unfolded)

print("\n")

w_flat = kernel.flatten()

yy = np.matmul(x_unfolded, w_flat)

yy = yy.reshape((mat_row - mask_size + 1, mat_col - mask_size + 1))

print(yy)
