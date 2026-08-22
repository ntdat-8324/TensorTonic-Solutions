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

class VisionTransformer:
    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 num_classes: int = 1000, embed_dim: int = 768,
                 depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0,
                 W_patch=None, cls_token=None, pos_embed=None,
                 encoder_weights=None, W_head=None):
        """
        Initialize Vision Transformer. If weight arrays are provided, use them;
        otherwise initialize randomly.
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes
        # Initialize weights here
        if W_patch is None:
            W_patch = np.random.randn(3*(patch_size*patch_size), embed_dim)
        self.W_patch = W_patch

        if cls_token is None:
            self.cls_token = np.random.randn(1, 1, self.embed_dim)
        else: self.cls_token = cls_token

        if pos_embed is None:
            self.pos_embed = np.random.randn(1, self.num_patches+1, self.embed_dim)*0.02
        else: self.pos_embed = pos_embed

        mlp_dim = int(embed_dim * mlp_ratio)
        
        if encoder_weights is not None:
            self.encoder_weights = encoder_weights
        else: 
            self.encoder_weights = []
            for _ in range(self.depth):
                layer_weights = {
                    # Multi-Head Attention
                    "Wq": np.random.randn(embed_dim, embed_dim) * 0.02,
                    "Wk": np.random.randn(embed_dim, embed_dim) * 0.02,
                    "Wv": np.random.randn(embed_dim, embed_dim) * 0.02,
                    "Wo": np.random.randn(embed_dim, embed_dim) * 0.02,
                    
                    # MLP
                    "W1": np.random.randn(embed_dim, mlp_dim) * 0.02,
                    "W2": np.random.randn(mlp_dim, embed_dim) * 0.02,
                }
                self.encoder_weights.append(layer_weights)

        if W_head is None:
            self.W_head = np.random.randn(embed_dim, self.num_classes) * 0.02
        else: self.W_head = W_head
    
    def encoder_block(self, x: np.ndarray, encoder_weight) -> np.ndarray:

    
        B, N, D = x.shape
    
        norm_x = layer_norm(x)
        
        q, k, v = norm_x @ encoder_weight['Wq'], norm_x @ encoder_weight['Wk'], norm_x @ encoder_weight['Wv']
        
        q = q.reshape(B, N, self.num_heads, D//self.num_heads).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, D//self.num_heads).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, D//self.num_heads).transpose(0, 2, 1, 3)
    
        scale = 1/np.sqrt(D//self.num_heads)
        attn_score = q@k.transpose(0, 1, 3, 2) #(B, H, N, N)
        attn_score = attn_score*scale
        attn_score = softmax(attn_score, axis=-1) 
    
        attn = attn_score@v # (B, H, N, N) @ (B, H, N, D) => (B, H, N, D) 
        attn = attn.transpose(0, 2, 1, 3).reshape(B, N, D)
        attn = attn@encoder_weight['Wo']
        x_ = x + attn
        attn_x = layer_norm(x_)
    
        attn_x = GELU(attn_x@encoder_weight['W1'])
        attn_x = attn_x@encoder_weight['W2']
        out = x_ + attn_x
        return out    
        

    def patch_embed(self, image: np.ndarray):
        B, H, W, C = image.shape

        H_p = self.patch_size
        W_p = self.patch_size

        img_patch = np.zeros((B, self.num_patches, H_p*W_p*C))

        for b in range(B):
            p = 0
            for h_p in range(H_p):
                for w_p in range(W_p):

                    h_start = h_p*self.patch_size
                    w_start = w_p*self.patch_size

                    h_end = h_start+self.patch_size
                    w_end = w_start+self.patch_size

                    img_patch[b, p, :] = image[b, h_start:h_end, w_start:w_end, :].flatten()
                    p+=1
        img_emb = img_patch@self.W_patch
        return img_emb

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        """
        # YOUR CODE HERE
        B = x.shape[0]
        
        img_emb = self.patch_embed(x) # (B, N, D)

        cls_token_b = np.tile(self.cls_token, (B, 1, 1))

        img_emb = np.concat((cls_token_b, img_emb), axis = 1) #(B, N+1, D)

        img_emb = img_emb + self.pos_embed

        for i in range(self.depth):
            img_emb = self.encoder_block(img_emb, self.encoder_weights[i])

        cls_emb = img_emb[:,0,:]
        cls_emb = layer_norm(cls_emb)
        logits = cls_emb@self.W_head
        
        return logits