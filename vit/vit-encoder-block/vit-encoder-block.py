import numpy as np

def layer_norm(x, eps=1e-6):
        mean = np.mean(x, axis=-1, keepdims=True) # mean.shape =  (B, N, 1)

        std = np.std(x, axis=-1, keepdims=True) # std.shape = (B, N, 1)

        x_norm = (x - mean) / (std + eps)
        
        return x_norm

def softmax (x, axis = None):
    # x.shape = (B, H, N, N)
    x_max = np.max(x, axis = axis, keepdims=True) # (B, H, N, 1); x_i < 0
    exp_x = np.exp(x-x_max) # (B, H, N, N) - (B, H, N, 1) => (B, H, N, N); exp_x_i > 0
    return exp_x / np.sum(exp_x, axis = axis, keepdims=True) # (B, H, N, N) - (B, H, N, 1) => (B, H, N, N)

def GELU(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def vit_encoder_block(x: np.ndarray, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                      Wq: np.ndarray = None, Wk: np.ndarray = None, Wv: np.ndarray = None,
                      Wo: np.ndarray = None, W1: np.ndarray = None, W2: np.ndarray = None) -> np.ndarray:
    """
    ViT Transformer encoder block with Pre-LayerNorm.
    Weight matrices are provided as inputs for deterministic testing.
    """
    # YOUR CODE HERE

    B, N, D = x.shape

    norm_x = layer_norm(x)

    if Wq is None: Wq = np.random.randn(D, D) * 0.02
    if Wk is None: Wk = np.random.randn(D, D) * 0.02
    if Wv is None: Wv = np.random.randn(D, D) * 0.02
    if Wo is None: Wo = np.random.randn(D, D) * 0.02
    if W1 is None: W1 = np.random.randn(D, D*mlp_ratio) * 0.02
    if W2 is None: W2 = np.random.randn(D*mlp_ratio, D) * 0.02
    
    q, k, v = norm_x @ Wq, norm_x @ Wk, norm_x @ Wv
    
    q = q.reshape(B, N, num_heads, D//num_heads).transpose(0, 2, 1, 3)
    k = k.reshape(B, N, num_heads, D//num_heads).transpose(0, 2, 1, 3)
    v = v.reshape(B, N, num_heads, D//num_heads).transpose(0, 2, 1, 3)

    scale = 1/np.sqrt(D//num_heads)
    attn_score = q@k.transpose(0, 1, 3, 2) #(B, H, N, N)
    attn_score = attn_score*scale
    attn_score = softmax(attn_score, axis=-1) 

    attn = attn_score@v # (B, H, N, N) @ (B, H, N, D) => (B, H, N, D) 
    attn = attn.transpose(0, 2, 1, 3).reshape(B, N, D)
    attn = attn@Wo
    x_ = x + attn
    attn_x = layer_norm(x_)

    attn_x = GELU(attn_x@W1)
    attn_x = attn_x@W2
    out = x_ + attn_x
    return out