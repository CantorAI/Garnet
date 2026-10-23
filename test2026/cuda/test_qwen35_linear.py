"""Native TensorRT parity for grouped Qwen3.5 state, BF16/F32, batches 1/2/4."""
import json,sys,os
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet
import tensor
from xModel.qwen35.vl_0_8b.linear_attention import T,Qwen35LinearAttentionStep
from xModel.qwen35.vl_0_8b.full_attention import Qwen35FullAttentionStep
from xModel.qwen35.vl_0_8b.vision_encoder import Qwen35VisionEncoder

with open(sys.argv[1]) as f:cases=json.loads(f.read())
def flattened(value):
    if isinstance(value,list):
        out=[]
        for item in value:out.extend(flattened(item))
        return out
    return [float(value)]
for case in cases:
    print(json.dumps({'event':'case_start','kind':case.get('kind','linear'),'dtype':case['dtype'],'batch':case['batch']}),flush=True)
    shapes=[a['shape'] for a in case['steps'][0]['inputs']]
    symbols=[tensor.input('linear_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
    @T.fusion(name='qwen35_cuda_linear_parity',boundary='required')
    def forward(*args):
        if case.get('kind')=='projection':return args[0]*T.unary_op('linear',weight_name=case['weight_name'],compute_dtype='bfloat16')
        if case.get('kind')=='vision':return Qwen35VisionEncoder(args[0],case['config'],case['grid_h'],case['grid_w'])
        if case.get('kind')=='attention':return Qwen35FullAttentionStep(*args,case['config'],0)
        return Qwen35LinearAttentionStep(*args,case['config'],0)
    graph=garnet.compiled_graph()
    try:
        graph.build(forward(*symbols),symbols,case['checkpoint'])
        for step in case['steps']:
            inputs=[garnet.tensor_from_host(a['values'],dtype='float32',shape=a['shape']) for a in step['inputs']]
            result=graph.run(inputs)
            actual=flattened(result.tolist());expected=step['expected']
            trace=os.environ.get('GARNET_PARITY_TRACE')
            if trace:
                with open(trace,'w') as f:f.write(json.dumps({'case':case,'step':step,'actual':actual}))
            assert len(actual)==len(expected),(len(actual),len(expected))
            absolute=2e-5
            relative=2e-4
            worst=0.
            for a,e in zip(actual,expected):
                assert abs(a-e)<=absolute+relative*abs(e),(case['dtype'],case['batch'],a,e)
                worst=max(worst,abs(a-e))
            print(json.dumps({'status':'pass','kind':case.get('kind','linear'),'dtype':case['dtype'],'batch':case['batch'],'max_abs_error':worst}),flush=True)
    except BaseException as error:
        print('CASE_FAILED',str(error),flush=True)
        raise
    finally:graph.close()
print('QWEN35_CUDA_PARITY_COMPLETE',len(cases),flush=True)
