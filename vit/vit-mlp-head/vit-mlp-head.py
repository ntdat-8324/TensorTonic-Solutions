import numpy as np

def layer_norm(x, eps=1e-6):
        mean = np.mean(x, axis=-1, keepdims=True) # mean.shape =  (B, N, 1)

        std = np.std(x, axis=-1, keepdims=True) # std.shape = (B, N, 1)

        x_norm = (x - mean) / (std + eps)
        
        return x_norm

def classification_head(encoder_output: np.ndarray, num_classes: int, W_head: np.ndarray = None) -> np.ndarray:
    """
    Classification head for ViT. Extract [CLS], LayerNorm, linear projection.
    W_head: projection matrix (D, num_classes). If None, initialize randomly.
    """
    # YOUR CODE HERE

    B, N, D = encoder_output.shape

    if W_head is None:
        W_head = np.random.randn((D, num_classes))
    
    h_cls = encoder_output[:, 0,:]
    h = layer_norm(h_cls)
    logits = h@W_head

    return logits