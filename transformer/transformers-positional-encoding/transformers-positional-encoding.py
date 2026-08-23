import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    pos_vec = np.arange(seq_length).reshape(-1, 1) # (N, 1)
    dim_vec = np.arange(0, d_model, 2) # (1, D)
    div_term = np.exp(dim_vec * (-np.log(10000.0) / d_model)) # (1, D)

    out_mat = np.zeros((seq_length, d_model))

    out_mat[:, 0:d_model+1:2] = np.sin(pos_vec * div_term)
    out_mat[:, 1:d_model:2] = np.cos(pos_vec * div_term)
    
    return out_mat