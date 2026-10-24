"""Real decoder-layer CPU parity and stateful CUDA replay checks; not a benchmark."""
import os,sys,json,math
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T,Qwen35LinearDecoderLayer,Qwen35FullDecoderLayer
checkpoint,oracle_path,output=sys.argv[1:4]
diagnostic=len(sys.argv)>4 and sys.argv[4]=='diagnostic'
with open(oracle_path) as f:oracle=json.load(f)
with open(os.path.join(checkpoint,'config.json')) as f:config=json.load(f)['text_config']
width=config['hidden_size'];reports=[]
def compare(actual,expected):
    assert len(actual)==len(expected) and all(math.isfinite(v) for v in actual)
    errors=[abs(a-b) for a,b in zip(actual,expected)]
    rel=(sum(e*e for e in errors)/max(1e-20,sum(v*v for v in expected)))**.5
    result={'max_absolute_error':max(errors),'relative_l2_error':rel,'passed':rel<=.01 and max(errors)<=.05}
    if not diagnostic:assert result['passed'],result
    return result
for item in oracle['layers']:
    graph=garnet.compiled_graph();io=garnet.cuda_tensor_io()
    try:
        kind=item['kind'];idx=item['layer'];s1shape=[1]+item['state_shapes'][0];s2shape=[1]+item['state_shapes'][1]
        shapes=[[1,width],s1shape,s2shape,[1]] if kind=='linear_attention' else [[1,width],s1shape,s2shape,[1,3],[1],[1]]
        symbols=[tensor.input('real_layer_'+str(idx)+'_'+str(i),shape=shape,dtype=tensor.float32) for i,shape in enumerate(shapes)]
        @T.fusion(name='actual_qwen38_layer',boundary='required')
        def forward(*args):
            if kind=='linear_attention':return Qwen35LinearDecoderLayer(*args,config,idx)
            return Qwen35FullDecoderLayer(*args,config,idx,oracle['capacity'])
        graph.build(forward(*symbols),symbols,checkpoint)
        args=[io.zeros(shape) for shape in shapes];io.update(args[-1],[1]);hidden=io.zeros([1,width]);cases=[]
        for step,case in enumerate(item['cases']):
            io.update(args[0],case['input'])
            if kind!='linear_attention':io.update(args[3],case['positions']);io.update(args[4],[step])
            checks=[]
            for enabled in [False,True,True,True,True]:
                graph.enable_cuda_graph(enabled);packed=graph.run_device(args);io.slice_into(packed,0,hidden)
                actual=garnet.tensor_to_cpu(hidden).tolist();checks.append(compare(actual,case['expected_hidden']))
            # Compare sampled state entries only after final replay; keep canonical GPU states for next token.
            offset=width;state_checks=[]
            for state,sample in zip(args[1:3],case['state_samples']):
                io.slice_into(packed,offset,state);values=garnet.tensor_to_cpu(state).tolist()
                state_checks.append(compare([values[i] for i in sample['indices']],sample['values']))
                offset+=len(values)
            cases.append({'hidden':checks,'sampled_states':state_checks})
        stats=json.loads(graph.stats_json());assert stats['graph_replays']>=2 and stats['graph_capture_failures']==0
        passed=all(c['passed'] for case in cases for c in case['hidden']+case['sampled_states'])
        reports.append({'layer':idx,'kind':kind,'cases':cases,'stats':stats,'passed':passed})
        with open(output,'w') as f:json.dump({'layers':reports,'completed':len(reports)==len(oracle['layers']),'scope':oracle['scope']},f,indent=2)
        print('ACTUAL_QWEN38_LAYER_CPU_PARITY_RESULT',idx,kind,'passed',passed)
    finally:graph.close();io.close()
with open(output,'w') as f:json.dump({'layers':reports,'scope':oracle['scope']},f,indent=2)
assert all(r['passed'] for r in reports),[(r['layer'],r['passed']) for r in reports]
