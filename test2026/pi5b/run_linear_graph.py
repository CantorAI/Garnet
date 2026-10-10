import garnet
import tensor
from xModel.qwen35.vl_0_8b import linear_attention as linear
config=garnet.config()
batch=config['batch']
channels=2*config['linear_num_key_heads']*config['linear_key_head_dim']+config['linear_num_value_heads']*config['linear_value_head_dim']
shapes=[[batch,config['hidden_size']],[batch,channels,config['linear_conv_kernel_dim']],
    [batch,config['linear_num_value_heads'],config['linear_key_head_dim'],config['linear_value_head_dim']],[batch]]
inputs=[tensor.input('linear_input_'+str(i),shape=shape,dtype=tensor.float32) for i,shape in enumerate(shapes)]
@linear.T.fusion(name='linear_fixture',boundary='required')
def forward(hidden,conv,state,active):
    packed=linear.Qwen35LinearAttentionStep(hidden,conv,state,active,config,0)
    if config['mode']=='packed':return packed
    output=linear.unpack(packed,config)
    if config['mode']=='out':return output[0]
    if config['mode']=='conv_state':return output[1]
    return output[2]
assert garnet.run_fixture(forward(*inputs),inputs)
