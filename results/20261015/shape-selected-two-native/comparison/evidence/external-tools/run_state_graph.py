import sys,json,math
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen35.vl_0_8b import gated_delta as delta,causal_conv as conv
with open(sys.argv[1]) as stream:f=json.load(stream)
symbols=[tensor.input('state_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(f['shapes'])]
component=delta if f['kind']=='delta' else conv
@component.T.fusion(name='metal_state_unpack_reference',boundary='required')
def capture(*inputs):
    value=delta.Qwen35GatedDeltaStep(*inputs) if f['kind']=='delta' else conv.Qwen35CausalConvStep(*inputs)
    if f['mode']=='packed':return value
    if f['kind']=='delta':
        return (delta.output_from_packed(value,7,5) if f['mode']=='output' else delta.state_from_packed(value,7,5))
    return (conv.output_from_packed(value,4) if f['mode']=='output' else conv.state_from_packed(value,4))
graph=garnet.compiled_graph()
try:
    graph.build(capture(*symbols),symbols,f['checkpoint'])
    for request in f['requests']:
        result=graph.run([tensor.tensor(data,shape=shape,dtype=tensor.float32)
            for data,shape in zip(request['inputs'],f['shapes'])])
        actual=result.tolist();expected=request[f['mode']]
        assert len(actual)==len(expected)
        for x,y in zip(actual,expected):
            assert math.isfinite(x) and abs(x-y)<=5e-6+5e-5*abs(y),(f['kind'],x,y)
    print(garnet.backend_info(),'captured',f['kind'],f['mode'],'graph batch',f['shapes'][0][0],
        'PASS requests',len(f['requests']),'run_ms',graph.last_inference_ms,flush=True)
finally:graph.close()
