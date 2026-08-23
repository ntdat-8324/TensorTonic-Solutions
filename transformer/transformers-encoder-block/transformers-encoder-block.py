import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # Your code here
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    norm = (x-mean)/(np.sqrt(var + eps))
    return gamma*norm + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # Your code here
    B, N, _ = Q.shape
    q, k, v = Q@W_q, K@W_k, V@W_v
    D = q.shape[-1]
    D_h = D // num_heads

    q = q.reshape(B, N, num_heads, D_h).transpose(0, 2, 1, 3)
    k = k.reshape(B, N, num_heads, D_h).transpose(0, 2, 3, 1)
    v = v.reshape(B, N, num_heads, D_h).transpose(0, 2, 1, 3)

    attn_score = softmax((q@k)*(1/np.sqrt(D_h)))
    attn_out = (attn_score@v)

    attn_out = attn_out.transpose(0,2,1,3).reshape(B, N, D)
    attn_out = attn_out@W_o
    return attn_out

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    x1 = x@W1 + b1
    x1 = np.clip(x1, min = 0)
    x2 = x1@W2 + b2
    return x2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here
    attn_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)

    x = attn_out + x
    x = layer_norm(x, gamma1, beta1)

    ffn_out = feed_forward(x, W1, b1, W2, b2)

    x = ffn_out + x
    x = layer_norm(x, gamma2, beta2)
    return x