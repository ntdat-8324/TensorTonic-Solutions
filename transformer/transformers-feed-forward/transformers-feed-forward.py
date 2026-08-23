import numpy as np

def ReLU (x: np.ndarray):
    return np.clip(x, min = 0)

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    # Your code here
    x_1 = ReLU(x@W1 + b1)
    x_2 = x_1@W2 + b2
    return x_2