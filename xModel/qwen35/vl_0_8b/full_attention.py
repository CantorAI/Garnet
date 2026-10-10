# SPDX-License-Identifier: Apache-2.0
"""Single-token gated full attention with explicit per-request KV buffers."""
import garnet
from .tensor_compat import tensor
T=tensor()
@T.fusion(name='qwen35_full_attention_step',role='attention',boundary='required')
def Qwen35FullAttentionStep(hidden,key_cache,value_cache,positions,slots,active,config,layer_idx):
    bound=hidden * T.binary_op('qwen35_attention_bind_key') * key_cache
    bound=bound * T.binary_op('qwen35_attention_bind_value') * value_cache
    bound=bound * T.binary_op('qwen35_attention_bind_positions') * positions
    bound=bound * T.binary_op('qwen35_attention_bind_slots') * slots
    bound=bound * T.binary_op('qwen35_attention_bind_active') * active
    rope=config['rope_parameters']
    return bound * T.unary_op('qwen35_attention_step',
        weight_prefix='model.language_model.layers.'+str(layer_idx)+'.self_attn',
        num_heads=config['num_attention_heads'],num_kv_heads=config['num_key_value_heads'],head_dim=config['head_dim'],
        rotary_dim=int(config['head_dim']*rope['partial_rotary_factor']),rope_theta=rope['rope_theta'],
        mrope_t=rope['mrope_section'][0],mrope_h=rope['mrope_section'][1],mrope_w=rope['mrope_section'][2],eps=config['rms_norm_eps'])
def unpack(packed,config,capacity):
    dimensions=dict(hidden_dim=config['hidden_size'],num_kv_heads=config['num_key_value_heads'],head_dim=config['head_dim'],capacity=capacity)
    return [packed * T.unary_op('qwen35_attention_output',**dimensions),
            packed * T.unary_op('qwen35_attention_key_cache',**dimensions),
            packed * T.unary_op('qwen35_attention_value_cache',**dimensions)]
