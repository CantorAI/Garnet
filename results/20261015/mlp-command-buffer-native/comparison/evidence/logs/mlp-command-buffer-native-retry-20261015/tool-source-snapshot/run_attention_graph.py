import sys,json,math
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen35.vl_0_8b import full_attention as attention
from xModel.qwen35.vl_0_8b.decoder_layer import Qwen35FullDecoderLayer
with open(sys.argv[1]) as stream:f=json.load(stream)
c=f['config'];b=c['batch'];cache=[b,c['num_key_value_heads'],c['capacity'],c['head_dim']]
shapes=[[b,c['hidden_size']],cache,cache,[b,3],[b],[b]]
symbols=[tensor.input('attention_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
@attention.T.fusion(name='metal_complete_attention_reference',boundary='required')
def forward(hidden,key,value,positions,slots,active):
    packed=(Qwen35FullDecoderLayer(hidden,key,value,positions,slots,active,c,3,c['capacity']) if c['decoder'] else
        attention.Qwen35FullAttentionStep(hidden,key,value,positions,slots,active,c,3))
    if c['mode']=='packed':return packed
    results=attention.unpack(packed,c,c['capacity'])
    if c['mode']=='out':return results[0]
    if c['mode']=='key':return results[1]
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
    # Reject invalid active slots before dispatching GPU kernels.
    first=f['steps'][0]
    if 'input_files' in first:
        bad=[garnet.tensor_io().read_f32(path,s) for path,s in zip(first['input_files'],shapes)]
    else:bad=[tensor.tensor(data,shape=s,dtype=tensor.float32) for data,s in zip(first['inputs'],shapes)]
    bad[4]=tensor.tensor([c['capacity']]*b,shape=[b],dtype=tensor.float32)
    bad[5]=tensor.tensor([1]*b,shape=[b],dtype=tensor.float32)
    rejected=False
    try:graph.run(bad)
    except Exception:rejected=True
    assert rejected,'Out-of-capacity active slot was accepted'
    print(garnet.backend_info(),'complete full','decoder' if c['decoder'] else 'attention','batch',b,
        'rotary',c['rope_parameters']['partial_rotary_factor'],'mode',c['mode'],
        'PASS steps',len(f['steps']),'run_ms',graph.last_inference_ms,flush=True)
finally:graph.close()
