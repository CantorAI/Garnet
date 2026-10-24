"""Real fused4/16 layers, state continuation, inactive padding and replay."""
import json,math,os,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet,tensor
from xModel.qwen35.vl_0_8b.cuda_chunk import T,Qwen35CUDAChunk
checkpoint,oracle_path,output=sys.argv[1:4]
chunk=int(sys.argv[4]) if len(sys.argv)>4 else 4
assert chunk in [4,16]
with open(oracle_path) as f:oracle=json.load(f)
with open(os.path.join(checkpoint,'config.json')) as f:config=json.load(f)['text_config']
width=config['hidden_size'];reports=[]
def compare(actual,expected):
    assert len(actual)==len(expected) and all(math.isfinite(x) for x in actual)
    errors=[abs(a-b) for a,b in zip(actual,expected)]
    rel=(sum(x*x for x in errors)/max(1e-20,sum(x*x for x in expected)))**.5
    row={'max_absolute_error':max(errors),'relative_l2_error':rel}
    assert rel<=.01 and row['max_absolute_error']<=.05,row
    return row
for item in oracle['layers']:
    graph=garnet.compiled_graph();io=garnet.cuda_tensor_io()
    try:
        kind=item['kind'];idx=item['layer'];s1=[1]+item['state_shapes'][0];s2=[1]+item['state_shapes'][1]
        shapes=[[1,chunk,width],s1,s2,[1,chunk]] if kind=='linear_attention' else [[1,chunk,width],s1,s2,[1,chunk,3],[1,chunk],[1,chunk]]
        symbols=[tensor.input('real_chunk_'+str(idx)+'_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
        @T.fusion(name='actual_qwen38_fused_chunk',boundary='required')
        def forward(*args):
            if kind=='linear_attention':return Qwen35CUDAChunk(args[0],args[1],args[2],args[3],None,None,config,idx,kind,True,True)
            return Qwen35CUDAChunk(args[0],args[1],args[2],args[5],args[3],args[4],config,idx,kind,True,True)
        graph.build(forward(*symbols),symbols,checkpoint);args=[io.zeros(s) for s in shapes];hidden=io.zeros([1,chunk,width]);chunks=[]
        for start,count in [(0,chunk),(chunk,2)]:
            selected=item['cases'][start:start+count]
            io.update(args[0],[v for c in selected for v in c['input']]+[0.]*(chunk-count)*width)
            io.update(args[-1],[1.]*count+[0.]*(chunk-count))
            if kind!='linear_attention':
                io.update(args[3],[v for c in selected for v in c['positions']]+[0.]*(chunk-count)*3)
                io.update(args[4],list(range(start,start+count))+[0]*(chunk-count))
            checks=[]
            for enabled in [False,True,True,True,True]:
                graph.enable_cuda_graph(enabled);packed=graph.run_device(args);io.slice_into(packed,0,hidden)
                actual=garnet.tensor_to_cpu(hidden).tolist()[:count*width]
                checks.append(compare(actual,[v for c in selected for v in c['expected_hidden']]))
            offset=chunk*width;states=[]
            for state,sample in zip(args[1:3],selected[-1]['state_samples']):
                io.slice_into(packed,offset,state);values=garnet.tensor_to_cpu(state).tolist()
                states.append(compare([values[i] for i in sample['indices']],sample['values']));offset+=len(values)
            chunks.append({'start':start,'active_tokens':count,'hidden':checks,'sampled_states':states})
        stats=json.loads(graph.stats_json());assert stats['graph_replays']>=2 and stats['graph_capture_failures']==0
        reports.append({'layer':idx,'kind':kind,'chunks':chunks,'stats':stats})
        with open(output,'w') as f:json.dump({'layers':reports,'completed':len(reports)==2,'tested_chunk':chunk,'tested_steps':chunk+2,'scope':oracle['scope']},f,indent=2)
        print('REAL_FUSED_QWEN38_CHUNK_PASS',idx,kind,flush=True)
    finally:graph.close();io.close()
