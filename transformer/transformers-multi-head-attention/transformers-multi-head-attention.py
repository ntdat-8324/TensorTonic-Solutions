import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    B, N, _ = Q.shape
    
    q, k, v = Q@W_q, K@W_k, V@W_v

    D = q.shape[-1]
    D_h = D // num_heads

    q = q.reshape(B, N, num_heads, D_h).transpose(0, 2, 1, 3)
    k = k.reshape(B, N, num_heads, D_h).transpose(0, 2, 3, 1)
    v = v.reshape(B, N, num_heads, D_h).transpose(0, 2, 1, 3)

    attn_score = softmax((q@k)*(1/np.sqrt(q.shape[-1])))

    attn_out = attn_score @ v # (B, h, N, D_h)

    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, N, D)
    attn_out = attn_out@W_o
    return attn_out