import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU activation: f(x) = max(0, x)
    """
    return np.clip(x, a_min=0, a_max=None)