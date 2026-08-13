import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def forget_gate(h_prev: np.ndarray, x_t: np.ndarray,
                W_f: np.ndarray, b_f: np.ndarray) -> np.ndarray:
    """Compute forget gate: f_t = sigmoid(W_f @ [h, x] + b_f)"""
    f_t = sigmoid(np.hstack((h_prev,x_t)) @ W_f.T + b_f) # h (B,D), x (B, d) => concat (B, D+d) ; W_f (D, H+D)
    
    return f_t