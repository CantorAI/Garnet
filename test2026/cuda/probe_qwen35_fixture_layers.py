"""Locate per-layer hidden error without modifying strict fixture checks."""
import json,os,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet
from xModel.qwen35.vl_0_8b.cuda_decoder import CUDADecoder
fixture=json.load(open(sys.argv[1]));output=sys.argv[2]
batch=fixture['batch'];hidden=fixture['config']['hidden_size'];chunk=4;observed={}
original=CUDADecoder._compile
def compiled(self,checkpoint,idx,kind,shapes):
    graph=original(self,checkpoint,idx,kind,shapes)
    class ProbeGraph:
        def __getattr__(self,name):return getattr(graph,name)
        def run_device(self,inputs):
            result=graph.run_device(inputs)
            values=garnet.tensor_to_cpu(result).tolist()
            assert not isinstance(values[0],list),'Expected flat native Tensor.tolist'
            stride=len(values)//batch
            observed[idx]=[values[b*stride:b*stride+chunk*hidden] for b in range(batch)]
            return result
    return ProbeGraph()
CUDADecoder._compile=compiled
model=CUDADecoder(fixture['checkpoint'],fixture['config'],batch,fixture['capacity'],chunk,True)
try:
    embeddings=[model.io.embedding(s['tokens']) for s in fixture['steps']]
    positions=[[] for b in range(batch)];slots=[0]*batch
    for step in fixture['steps']:
        for b in range(batch):positions[b].append([slots[b]]*3);slots[b]+=step['active'][b]
    model.chunk_hidden(embeddings,[s['active'] for s in fixture['steps']],positions)
    records=[]
    for layer,rows in observed.items():
        for b,row in enumerate(rows):
            for t,step in enumerate(fixture['steps']):
                expected=step['layer_hidden'][layer][b*hidden:(b+1)*hidden]
                actual=row[t*hidden:(t+1)*hidden]
                records.append({'layer':layer,'row':b,'step':t,'active':step['active'][b],
                    'actual':actual,'expected':expected,'max_abs':max(abs(x-y) for x,y in zip(actual,expected))})
    with open(output,'w') as stream:json.dump({'records':records,'limitation':'Diagnostic per-layer host copies invalidate timing; serial CPU reference geometry'},stream,indent=2)
    print('QWEN35_FIXTURE_LAYER_PROBE_COMPLETE',len(records),flush=True)
finally:model.close()
