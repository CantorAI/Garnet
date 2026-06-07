import CpuTensor as T

def CrossAttention(x, context, dim, num_heads):
    head_dim = dim / num_heads
    
    q = x * T.matmul() * weights_q + bias_q
    k = context * T.matmul() * weights_k + bias_k
    v = context * T.matmul() * weights_v + bias_v
    
    q = q * T.reshape([query_seq_len, num_heads, head_dim]) * T.permute([1, 0, 2])
    k = k * T.reshape([kv_seq_len, num_heads, head_dim]) * T.permute([1, 0, 2])
    v = v * T.reshape([kv_seq_len, num_heads, head_dim]) * T.permute([1, 0, 2])
    
    scale = head_dim ** -0.5
    attn = q * T.matmul() * (k * T.permute([0, 2, 1])) * scale
    attn_probs = attn * T.softmax(axis=-1)
    
    out = attn_probs * T.matmul() * v
    
    out = out * T.permute([1, 0, 2]) * T.reshape([query_seq_len, dim])
    out = out * T.matmul() * weights_proj + bias_proj
    
    return out

@fusion
def VisionLanguageAdapter(vision_features, dim, num_heads):
    # Initialize learned queries
    query_tokens = query_embeddings 
    
    # Apply LayerNorm
    q_norm = query_tokens * T.layer_norm(axis=-1)
    v_norm = vision_features * T.layer_norm(axis=-1)
    
    # Incorporate 2D absolute positional embeddings into queries
    q_norm = q_norm + query_pos_embeddings
    
    # Cross Attention pooling
    attn_out = CrossAttention(q_norm, v_norm, dim, num_heads)
    
    # Residual
    x = query_tokens + attn_out
    
    # MLP projection
    norm_x = x * T.layer_norm(axis=-1)
    h = norm_x * T.matmul() * mlp_weights_1 + mlp_bias_1
    h = h * T.gelu()
    out = h * T.matmul() * mlp_weights_2 + mlp_bias_2
    
    # Final residual
    out = x + out
    
    return out
