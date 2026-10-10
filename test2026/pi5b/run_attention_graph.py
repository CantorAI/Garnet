import garnet
import tensor
from xModel.qwen35.vl_0_8b import full_attention as attention
config=garnet.config();batch=config['batch'];kv=config['num_key_value_heads'];dim=config['head_dim'];capacity=config['capacity']
shapes=[[batch,config['hidden_size']],[batch,kv,capacity,dim],[batch,kv,capacity,dim],[batch,3],[batch],[batch]]
inputs=[tensor.input('attention_input_'+str(i),shape=shape,dtype=tensor.float32) for i,shape in enumerate(shapes)]
@attention.T.fusion(name='attention_fixture',boundary='required')
def forward(hidden,key,value,positions,slots,active):
    packed=attention.Qwen35FullAttentionStep(hidden,key,value,positions,slots,active,config,config['layer_idx'])
    if config['mode']=='packed':return packed
    output=attention.unpack(packed,config,capacity)
    if config['mode']=='out':return output[0]
    if config['mode']=='key_cache':return output[1]
    return output[2]
assert garnet.run_fixture(forward(*inputs),inputs)
