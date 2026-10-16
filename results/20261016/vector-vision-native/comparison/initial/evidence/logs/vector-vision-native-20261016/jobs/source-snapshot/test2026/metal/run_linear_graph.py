import sys,json,math
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen35.vl_0_8b import linear_attention as linear
from xModel.qwen35.vl_0_8b.decoder_layer import Qwen35LinearDecoderLayer
with open(sys.argv[1]) as stream:f=json.load(stream)
c=f['config'];batch=c['batch']
channels=2*c['linear_num_key_heads']*c['linear_key_head_dim']+c['linear_num_value_heads']*c['linear_value_head_dim']
shapes=[[batch,c['hidden_size']],[batch,channels,c['linear_conv_kernel_dim']],
    [batch,c['linear_num_value_heads'],c['linear_key_head_dim'],c['linear_value_head_dim']],[batch]]
symbols=[tensor.input('linear_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
@linear.T.fusion(name='metal_complete_linear_reference',boundary='required')
def forward(hidden,conv,state,active):
    packed=(Qwen35LinearDecoderLayer(hidden,conv,state,active,c,0) if c['decoder'] else
        linear.Qwen35LinearAttentionStep(hidden,conv,state,active,c,0))
    if c['mode']=='packed':return packed
    results=linear.unpack(packed,c)
    if c['mode']=='out':return results[0]
    if c['mode']=='conv_state':return results[1]
    return results[2]
graph=garnet.compiled_graph()
try:
    graph.build(forward(*symbols),symbols,f['checkpoint'])
    for step in f['steps']:
        if 'input_files' in step:
            inputs=[garnet.tensor_io().read_f32(path,s) for path,s in zip(step['input_files'],shapes)]
            expected=garnet.tensor_io().read_f32(step['expected_file'],[step['expected_count']]).tolist()
        else:
            inputs=[tensor.tensor(data,shape=s,dtype=tensor.float32) for data,s in zip(step['inputs'],shapes)]
            expected=step['expected']
        actual=graph.run(inputs).tolist()
        assert len(actual)==len(expected)
        for x,y in zip(actual,expected):
            assert math.isfinite(x) and abs(x-y)<=5e-6+5e-5*abs(y),(c,x,y)
    print(garnet.backend_info(),'complete linear','decoder' if c['decoder'] else 'attention',
        'batch',batch,'heads',c['linear_num_key_heads'],c['linear_num_value_heads'],c['mode'],
        'PASS steps',len(f['steps']),'run_ms',graph.last_inference_ms,flush=True)
finally:graph.close()
