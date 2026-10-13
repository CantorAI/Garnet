# SPDX-License-Identifier: Apache-2.0
"""Fixed-size prefill: batched normalization/MLP, causal sequential attention."""
from .decoder_layer import T,norm,mlp,add
from .linear_attention import Qwen35LinearAttentionStep,unpack as linear_unpack
from .full_attention import Qwen35FullAttentionStep,unpack as full_unpack

def reshape(value,shape):return value * T.unary_op('reshape',shape=shape)
def token(value,index):return value * T.unary_op('qwen35_chunk_token',index=index)
def concat(left,right):return left * T.binary_op('qwen35_concat_sequence') * right

@T.fusion(name='qwen35_prefill_layer',role='decoder_layer',boundary='required')
def PrefillLayer(hidden,state1,state2,positions,slots,active,config,layer_idx,batch,chunk,capacity,kind,parallel_attention=False):
    width=config['hidden_size'];prefix='model.language_model.layers.'+str(layer_idx)
    hidden=hidden * T.binary_op('qwen35_mask_chunk') * active
    normalized=reshape(norm(reshape(hidden,[batch*chunk,width]),prefix+'.input_layernorm',config),[batch,chunk,width])
    outputs=None
    if parallel_attention and kind!='linear_attention' and capacity<=2048:
        packed=Qwen35FullAttentionStep(normalized,state1,state2,positions,slots,active,config,layer_idx,chunk=True)
        packed_config=dict(config);packed_config['hidden_size']=chunk*width
        outputs,state1,state2=full_unpack(packed,packed_config,capacity)
        outputs=reshape(outputs,[batch,chunk,width])
    for i in range(chunk if outputs is None else 0):
        if kind=='linear_attention':
            packed=Qwen35LinearAttentionStep(token(normalized,i),state1,state2,token(active,i),config,layer_idx)
            output,state1,state2=linear_unpack(packed,config)
        else:
            packed=Qwen35FullAttentionStep(token(normalized,i),state1,state2,token(positions,i),token(slots,i),token(active,i),config,layer_idx)
            output,state1,state2=full_unpack(packed,config,capacity)
        row=reshape(output,[batch,1,width])
        outputs=row if outputs is None else concat(outputs,row)
    residual=reshape(add(hidden,outputs),[batch*chunk,width])
    result=add(residual,mlp(norm(residual,prefix+'.post_attention_layernorm',config),prefix+'.mlp'))
    return concat(concat(reshape(result,[batch,chunk*width]),reshape(state1,[batch,-1])),reshape(state2,[batch,-1]))
