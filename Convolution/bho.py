import numpy as np
from scipy import linalg

h = np.arange(1, 6)

padding = np.zeros(h.shape[0] - 1, h.dtype)
first_col = np.r_[h, padding]
first_row = np.r_[h[0], padding]

H = linalg.toeplitz(first_col, first_row)

print(repr(H))
# array([[1, 0, 0, 0, 0],
#        [2, 1, 0, 0, 0],
#        [3, 2, 1, 0, 0],
#        [4, 3, 2, 1, 0],
#        [5, 4, 3, 2, 1],
#        [0, 5, 4, 3, 2],
#        [0, 0, 5, 4, 3],
#        [0, 0, 0, 5, 4],
#        [0, 0, 0, 0, 5]])