import garnet
import tensor
from xModel.qwen35.vl_0_8b import decoder_layer as decoder
config=garnet.config();batch=config['batch']
if config['kind']=='linear':
    channels=2*config['linear_num_key_heads']*config['linear_key_head_dim']+config['linear_num_value_heads']*config['linear_value_head_dim']
    shapes=[[batch,config['hidden_size']],[batch,channels,config['linear_conv_kernel_dim']],
        [batch,config['linear_num_value_heads'],config['linear_key_head_dim'],config['linear_value_head_dim']],[batch]]
else:
    kv=config['num_key_value_heads'];dim=config['head_dim'];capacity=config['capacity']
    shapes=[[batch,config['hidden_size']],[batch,kv,capacity,dim],[batch,kv,capacity,dim],[batch,3],[batch],[batch]]
inputs=[tensor.input('decoder_input_'+str(i),shape=shape,dtype=tensor.float32) for i,shape in enumerate(shapes)]
@decoder.T.fusion(name='decoder_fixture',boundary='required')
def forward(*args):
    if config['kind']=='linear':return decoder.Qwen35LinearDecoderLayer(*args,config,config['layer_idx'])
    return decoder.Qwen35FullDecoderLayer(*args,config,config['layer_idx'],config['capacity'])
assert garnet.run_fixture(forward(*inputs),inputs)
