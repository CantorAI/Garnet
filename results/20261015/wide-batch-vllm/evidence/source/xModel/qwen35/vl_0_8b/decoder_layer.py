# SPDX-License-Identifier: Apache-2.0
"""Complete single-token decoder blocks with explicit caller-owned states."""
import garnet
from .tensor_compat import tensor
from .linear_attention import Qwen35LinearAttentionStep,unpack as unpack_linear
from .full_attention import Qwen35FullAttentionStep,unpack as unpack_attention
T=tensor()
def norm(hidden,prefix,config):
    return hidden * T.unary_op('qwen35_rms_norm',weight_name=prefix+'.weight',eps=config['rms_norm_eps'])
def mlp(hidden,prefix):
    return hidden * T.unary_op('qwen35_mlp',weight_prefix=prefix)
def add(left,right):
    return left * T.binary_op('add') * right
def replace(packed,hidden,config):
    return packed * T.binary_op('qwen35_replace_packed_hidden',hidden_dim=config['hidden_size']) * hidden

@T.fusion(name='qwen35_linear_decoder_layer',role='decoder_layer',boundary='required')
def Qwen35LinearDecoderLayer(hidden,conv,state,active,config,layer_idx):
    prefix='model.language_model.layers.'+str(layer_idx)
    hidden=hidden * T.binary_op('qwen35_mask_rows') * active
    normalized=norm(hidden,prefix+'.input_layernorm',config)
    packed=Qwen35LinearAttentionStep(normalized,conv,state,active,config,layer_idx)
    attention=unpack_linear(packed,config)[0]
    residual=add(hidden,attention)
    normalized=norm(residual,prefix+'.post_attention_layernorm',config)
    return replace(packed,add(residual,mlp(normalized,prefix+'.mlp')),config)

@T.fusion(name='qwen35_full_decoder_layer',role='decoder_layer',boundary='required')
def Qwen35FullDecoderLayer(hidden,key,value,positions,slots,active,config,layer_idx,capacity):
    prefix='model.language_model.layers.'+str(layer_idx)
    hidden=hidden * T.binary_op('qwen35_mask_rows') * active
    normalized=norm(hidden,prefix+'.input_layernorm',config)
    packed=Qwen35FullAttentionStep(normalized,key,value,positions,slots,active,config,layer_idx)
    attention=unpack_attention(packed,config,capacity)[0]
    residual=add(hidden,attention)
    normalized=norm(residual,prefix+'.post_attention_layernorm',config)
    return replace(packed,add(residual,mlp(normalized,prefix+'.mlp')),config)
