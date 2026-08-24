import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    n_r = len(A)
    n_c = len(A[0])
    out = np.zeros((n_c, n_r))
    for i in range(n_c):
        for j in range(n_r):
            out[i,j] = A[j][i]
    return out
