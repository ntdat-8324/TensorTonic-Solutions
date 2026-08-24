import numpy as np

def matrix_trace(A: list) -> float:
    """Return the sum of the main diagonal."""
    # Write code here
    n = len(A)
    c = 0
    for i in range(n):
        c+=A[i][i]
    return c