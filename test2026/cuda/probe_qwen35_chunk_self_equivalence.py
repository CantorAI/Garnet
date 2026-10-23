"""Native multi-token versus padded serial execution; independent oracle gates stay separate."""
import json,os,struct,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet
from xModel.qwen35.vl_0_8b.cuda_decoder import CUDADecoder
fixture_path,output=sys.argv[1:3];chunk=int(sys.argv[3]) if len(sys.argv)>3 else 16
fixture=json.load(open(fixture_path));b=fixture['batch'];h=fixture['config']['hidden_size']
model=CUDADecoder(fixture['checkpoint'],fixture['config'],b,fixture['capacity'],chunk,True,True)
def cpu(t):return garnet.tensor_to_cpu(t).tolist()
def bits(v):return struct.pack('<'+str(len(v))+'f',*v)
try:
    model.enable_cuda_graph(True);embedded=[model.io.embedding(s['tokens']) for s in fixture['steps']]
    positions=[[] for _ in range(b)];slots=[0]*b
    for step in fixture['steps']:
        for row in range(b):positions[row].append([slots[row]]*3);slots[row]+=step['active'][row]
    model.chunk_hidden(embedded,[s['active'] for s in fixture['steps']],positions)
    full=cpu(model.hidden);full_states=[cpu(layer[key]) for layer in model.layers for key in ('state1','state2')]
    model.reset();records=[]
    for token_index,step in enumerate(fixture['steps']):
        model.step(step['tokens'],step['active']);serial=cpu(model.hidden)
        for row in range(b):
            a=full[(row*chunk+token_index)*h:(row*chunk+token_index+1)*h]
            e=serial[row*chunk*h:row*chunk*h+h]
            records.append({'row':row,'token_index':token_index,'active':step['active'][row],
                'bitwise_equal':bits(a)==bits(e),'max_abs':max(abs(x-y) for x,y in zip(a,e)),
                'multi_token':a,'serial':e})
    states=[]
    for index,(a,e) in enumerate(zip(full_states,[cpu(layer[key]) for layer in model.layers for key in ('state1','state2')])):
        states.append({'index':index,'bitwise_equal':bits(a)==bits(e),'max_abs':max(abs(x-y) for x,y in zip(a,e))})
    result={'fixture':fixture_path,'batch':b,'chunk':chunk,'records':records,'states':states,
        'all_bitwise_equal':all(r['bitwise_equal'] for r in records+states),
        'scope':'Native execution-arrangement diagnostic, not independent mathematical correctness, pretrained verification, commit/rollback or speculative serving acceptance'}
    with open(output,'w') as stream:json.dump(result,stream,indent=2)
    print('QWEN35_NATIVE_CHUNK_SELF_EQUIVALENCE',result['all_bitwise_equal'],'batch',b,'chunk',chunk,flush=True)
    assert result['all_bitwise_equal'],'Multi-token/native serial discrepancy; inspect saved output'
finally:model.close()
