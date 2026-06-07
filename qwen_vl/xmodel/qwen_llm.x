import CpuTensor as T

def LlamaAttention(x, dim, num_heads, kv_heads, position_ids):
    head_dim = dim / num_heads
    
    q = x * T.matmul() * weights_q + bias_q
    k = x * T.matmul() * weights_k + bias_k
    v = x * T.matmul() * weights_v + bias_v
    
    q = q * T.reshape([seq_len, num_heads, head_dim]) * T.permute([1, 0, 2])
    k = k * T.reshape([seq_len, kv_heads, head_dim]) * T.permute([1, 0, 2])
    v = v * T.reshape([seq_len, kv_heads, head_dim]) * T.permute([1, 0, 2])
    
    # Apply Rotary Positional Embeddings (RoPE)
    q = q * T.rope(position_ids)
    k = k * T.rope(position_ids)
    
    scale = head_dim ** -0.5
    attn = q * T.matmul() * (k * T.permute([0, 2, 1])) * scale
    
    # Causal Masking
    attn = attn + causal_mask
    
    attn_probs = attn * T.softmax(axis=-1)
    
    out = attn_probs * T.matmul() * v
    
    out = out * T.permute([1, 0, 2]) * T.reshape([seq_len, dim])
    out = out * T.matmul() * weights_proj
    
    return out

def SwiGLU_MLP(x, in_dim, hidden_dim):
    # SwiGLU activation
    gate = x * T.matmul() * weights_gate
    up = x * T.matmul() * weights_up
    
    h = gate * T.silu() * up
    out = h * T.matmul() * weights_down
    
    return out

def QWenDecoderBlock(x, dim, num_heads, kv_heads, mlp_hidden_dim, position_ids):
    # Pre-normalization with RMSNorm
    norm_x = x * T.rms_norm(axis=-1)
    attn_out = LlamaAttention(norm_x, dim, num_heads, kv_heads, position_ids)
    x = x + attn_out
    
    norm_x = x * T.rms_norm(axis=-1)
    mlp_out = SwiGLU_MLP(norm_x, dim, mlp_hidden_dim)
    x = x + mlp_out
    
    return x

@fusion
def QWenLLM(input_embeddings, num_blocks, dim, num_heads, kv_heads, mlp_hidden_dim, position_ids):
    x = input_embeddings
    
    for i in range(num_blocks):
        x = QWenDecoderBlock(x, dim, num_heads, kv_heads, mlp_hidden_dim, position_ids)
        
    x = x * T.rms_norm(axis=-1)
    logits = x * T.matmul() * lm_head_weights
    
    return logits
