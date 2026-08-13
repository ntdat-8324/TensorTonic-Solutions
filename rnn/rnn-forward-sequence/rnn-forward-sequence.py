import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    batch_size = X.shape[0]
    tok_num = X.shape[1]
    hidden_dim = h_0.shape[1]

    hidden_states = np.zeros((batch_size, tok_num, hidden_dim))
    hid_cur = h_0
    for i in range(tok_num):

        x_cur = X[:,i,:]

        hidden_states[:,i,:] = np.tanh(x_cur@W_xh.T + hid_cur@W_hh.T + b_h)

        hid_cur = hidden_states[:,i,:]

    return hidden_states, hidden_states[:, -1, :]