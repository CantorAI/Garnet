# SPDX-License-Identifier: Apache-2.0
"""Bounded CUDA decoder chunk, also used for padded single-token decode.

Each layer has one compiled engine/weight set. BF16 input/output projections
and MLPs span the chunk; recurrent state evolves in token order on the GPU.
The existing Metal and serial TE programs are unchanged.
"""
from .tensor_compat import tensor
T=tensor()

@T.fusion(name='qwen35_cuda_decoder_chunk',role='decoder_layer',boundary='required')
def Qwen35CUDAChunk(hidden,state1,state2,active,positions,slots,config,layer_idx,kind,fused_delta=False,fused_attention=False,projection_workspace_bytes=0,retain_linear_inputs=False,external_weights=None,register_delta_128=False,join_input_projections=False):
    p='model.language_model.layers.'+str(layer_idx)
    extra={'external_weights':external_weights} if external_weights else {}
    if kind=='linear_attention':
        bound=hidden*T.binary_op('qwen35_conv_bind_state')*state1
        bound=bound*T.binary_op('qwen35_delta_bind_state')*state2
        bound=bound*T.binary_op('qwen35_conv_bind_active')*active
        return bound*T.unary_op('qwen35_linear_decoder_chunk',layer_prefix=p,
            key_heads=config['linear_num_key_heads'],value_heads=config['linear_num_value_heads'],
            key_dim=config['linear_key_head_dim'],value_dim=config['linear_value_head_dim'],eps=config['rms_norm_eps'],fused_delta=int(fused_delta),projection_workspace_bytes=projection_workspace_bytes,retain_linear_inputs=int(retain_linear_inputs),register_delta_128=int(register_delta_128),joined_input_projections=int(join_input_projections),**extra)
    bound=hidden*T.binary_op('qwen35_attention_bind_key')*state1
    bound=bound*T.binary_op('qwen35_attention_bind_value')*state2
    bound=bound*T.binary_op('qwen35_attention_bind_positions')*positions
    bound=bound*T.binary_op('qwen35_attention_bind_slots')*slots
    bound=bound*T.binary_op('qwen35_attention_bind_active')*active
    rope=config['rope_parameters']
    return bound*T.unary_op('qwen35_full_decoder_chunk',layer_prefix=p,
        num_heads=config['num_attention_heads'],num_kv_heads=config['num_key_value_heads'],head_dim=config['head_dim'],
        rotary_dim=int(config['head_dim']*rope['partial_rotary_factor']),rope_theta=rope['rope_theta'],
        mrope_t=rope['mrope_section'][0],mrope_h=rope['mrope_section'][1],mrope_w=rope['mrope_section'][2],eps=config['rms_norm_eps'],fused_attention=int(fused_attention),projection_workspace_bytes=projection_workspace_bytes,**extra)
