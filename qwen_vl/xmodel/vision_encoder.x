import CpuTensor as T

def Attention(x, dim, num_heads):
    head_dim = dim / num_heads
    
    # Linear projections for Q, K, V
    qkv = x * T.matmul() * weights_qkv + bias_qkv
    
    # Split into q, k, v
    q, k, v = qkv * T.split([dim, dim, dim], axis=-1)
    
    # Reshape for multi-head attention
    q = q * T.reshape([seq_len, num_heads, head_dim]) * T.permute([1, 0, 2])
    k = k * T.reshape([seq_len, num_heads, head_dim]) * T.permute([1, 0, 2])
    v = v * T.reshape([seq_len, num_heads, head_dim]) * T.permute([1, 0, 2])
    
    # Scaled dot-product attention
    scale = head_dim ** -0.5
    attn = q * T.matmul() * (k * T.permute([0, 2, 1])) * scale
    attn_probs = attn * T.softmax(axis=-1)
    
    out = attn_probs * T.matmul() * v
    
    # Combine heads and output projection
    out = out * T.permute([1, 0, 2]) * T.reshape([seq_len, dim])
    out = out * T.matmul() * weights_proj + bias_proj
    
    return out

def MLP(x, in_dim, hidden_dim):
    # MLP with GELU activation
    h = x * T.matmul() * weights_fc1 + bias_fc1
    h = h * T.gelu()
    out = h * T.matmul() * weights_fc2 + bias_fc2
    return out

def TransformerBlock(x, dim, num_heads, mlp_hidden_dim):
    # Pre-normalization architecture
    norm_x = x * T.layer_norm(axis=-1)
    attn_out = Attention(norm_x, dim, num_heads)
    x = x + attn_out
    
    norm_x = x * T.layer_norm(axis=-1)
    mlp_out = MLP(norm_x, dim, mlp_hidden_dim)
    x = x + mlp_out
    
    return x

@fusion
def VisionEncoder(pixel_values, num_blocks, dim, num_heads, mlp_hidden_dim):
    # 1. Patch Embedding (Conv2D equivalent via reshape and matmul)
    x = pixel_values * T.patch_embed(patch_size=14, stride=14) 
    
    # Add absolute positional embeddings
    x = x + position_embeddings
    
    # 2. Transformer Blocks
    for i in range(num_blocks):
        x = TransformerBlock(x, dim, num_heads, mlp_hidden_dim)
        
    return x
