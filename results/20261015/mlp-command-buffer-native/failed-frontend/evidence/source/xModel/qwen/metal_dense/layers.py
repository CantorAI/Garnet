# SPDX-License-Identifier: Apache-2.0
"""Dense Qwen3 and Qwen2.5 decoder graphs for native Garnet Metal."""
from xModel.qwen35.vl_0_8b.tensor_compat import tensor
from xModel.qwen35.vl_0_8b.decoder_layer import add,mlp,replace
from xModel.qwen35.vl_0_8b.full_attention import unpack
T=tensor()
def token(value,index):return value * T.unary_op("qwen35_chunk_token",index=index)
def reshape(value,shape):return value * T.unary_op("reshape",shape=shape)

def norm(hidden,prefix,config):
    return hidden * T.unary_op('qwen_text_norm',weight_name=prefix+'.weight',eps=config['rms_norm_eps'])

def attention(hidden,key,value,positions,slots,active,config,idx,chunk=False):
    bound=hidden * T.binary_op('qwen35_attention_bind_key') * key
    bound=bound * T.binary_op('qwen35_attention_bind_value') * value
    bound=bound * T.binary_op('qwen35_attention_bind_positions') * positions
    bound=bound * T.binary_op('qwen35_attention_bind_slots') * slots
    bound=bound * T.binary_op('qwen35_attention_bind_active') * active
    dim=config['head_dim']
    return bound * T.unary_op('qwen_text_attention_chunk' if chunk else 'qwen_text_attention_step',weight_prefix='model.layers.'+str(idx)+'.self_attn',
        num_heads=config['num_attention_heads'],num_kv_heads=config['num_key_value_heads'],head_dim=dim,
        rotary_dim=dim,mrope_t=dim//2,mrope_h=0,mrope_w=0,rope_theta=config['rope_theta'],eps=config['rms_norm_eps'],
        qk_norm=1 if config['model_type']=='qwen3' else 0,attention_bias=1 if config['model_type']=='qwen2' else 0)

def DenseLayer(hidden,key,value,positions,slots,active,config,idx,capacity):
    prefix='model.layers.'+str(idx)
    hidden=hidden * T.binary_op('qwen35_mask_rows') * active
    packed=attention(norm(hidden,prefix+'.input_layernorm',config),key,value,positions,slots,active,config,idx)
    residual=add(hidden,unpack(packed,config,capacity)[0])
    return replace(packed,add(residual,mlp(norm(residual,prefix+'.post_attention_layernorm',config),prefix+'.mlp')),config)

def DensePrefillLayer(hidden,key,value,positions,slots,active,config,idx,batch,chunk,capacity,kind):
    width=config['hidden_size'];prefix='model.layers.'+str(idx)
    hidden=hidden * T.binary_op('qwen35_mask_chunk') * active
    normalized=reshape(norm(reshape(hidden,[batch*chunk,width]),prefix+'.input_layernorm',config),[batch,chunk,width])
    outputs=None
    if capacity<=2048:
        packed=attention(normalized,key,value,positions,slots,active,config,idx,chunk=True)
        packed_config=dict(config);packed_config['hidden_size']=chunk*width
        outputs,key,value=unpack(packed,packed_config,capacity)
        outputs=reshape(outputs,[batch,chunk,width])
    else:
        for i in range(chunk):
            packed=attention(token(normalized,i),key,value,token(positions,i),token(slots,i),token(active,i),config,idx)
            output,key,value=unpack(packed,config,capacity)
            output=reshape(output,[batch,1,width])
            outputs=output if outputs is None else outputs * T.binary_op('qwen35_concat_sequence') * output
    residual=reshape(add(hidden,outputs),[batch*chunk,width])
    result=add(residual,mlp(norm(residual,prefix+'.post_attention_layernorm',config),prefix+'.mlp'))
    result=reshape(result,[batch,chunk*width])
    result=result * T.binary_op('qwen35_concat_sequence') * reshape(key,[batch,-1])
    return result * T.binary_op('qwen35_concat_sequence') * reshape(value,[batch,-1])
