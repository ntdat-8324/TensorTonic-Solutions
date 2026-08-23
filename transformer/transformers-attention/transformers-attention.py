import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    dim = Q.shape[-1]
    attn = torch.matmul(Q, K.transpose(-2, -1)) * (1/math.sqrt(dim))
    attn_score = torch.softmax(attn, axis=-1)

    attn_out = torch.matmul(attn_score, V)

    return attn_out