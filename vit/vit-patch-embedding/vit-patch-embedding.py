import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int, W_proj: np.ndarray = None) -> np.ndarray:
    """
    Convert image to patch embeddings.
    W_proj: projection matrix of shape (patch_dim, embed_dim). If None, initialize randomly.
    """
    # YOUR CODE HERE
    #image (B, H, W, C)
    #patch_size = PxP => patch (B, N, P^2*C)
    #output (B, N, D)

    batch_size , height , weight, channels = image.shape

    h_num = int(height // patch_size)
    w_num = int(weight // patch_size)

    patch_num = h_num*w_num

    patchs = np.zeros((batch_size, patch_num, patch_size*patch_size*channels))

    if W_proj is None:
        std = np.sqrt(2.0 / embed_dim)
        W_proj = np.random.normal(loc=0.0, scale=std, size=((patch_size*patch_size*channels), embed_dim))

    output = np.zeros((batch_size, patch_num, embed_dim))

    for b in range(batch_size):
        p = 0
        for i in range(h_num):
            for j in range(w_num):

                h_start = i*patch_size
                w_start = j*patch_size
    
                h_end = h_start + patch_size
                w_end = w_start + patch_size
    
                patchs[b, p,:] = image[b, h_start:h_end, w_start:w_end, :].flatten()
                p+=1
    output = patchs@W_proj
    return  output