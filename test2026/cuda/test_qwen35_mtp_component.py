"""Independent MTP hidden/KV checks, inactive/input preservation and graph replay."""
import json,os,struct,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet
from xModel.qwen35.vl_0_8b.cuda_mtp import CUDAMTPProgram
def read(spec):
    data=open(spec['path'],'rb').read();return list(struct.unpack('<'+str(len(data)//4)+'f',data))
def cpu(t):return garnet.tensor_to_cpu(t).tolist()
collect='--collect-failures' in sys.argv[2:];failures=[]
def compare(actual,expected,context):
    assert len(actual)==len(expected)
    error=max(abs(x-y) for x,y in zip(actual,expected))
    bad=[(i,x,y) for i,(x,y) in enumerate(zip(actual,expected)) if abs(x-y)>2e-5+2e-4*abs(y)]
    if bad:failures.append({'max_abs':error,'context':context,'first_bad':bad[:8],'count':len(bad)})
    if not collect:assert not bad,('MTP max_abs',error,'context',context,'first_bad',bad[:8])
    return error
cases=json.load(open(sys.argv[1]));overall=0
for case_index,c in enumerate(cases):
    initial_failures=len(failures)
    io=garnet.cuda_tensor_io();io.open(c['checkpoint']);b=c['batch'];t=c['chunk'];h=c['config']['hidden_size']
    state_shape=[b,c['config']['num_key_value_heads'],c['capacity'],c['config']['head_dim']]
    key=io.zeros(state_shape);value=io.zeros(state_shape);hidden=io.zeros([b,t,h]);state_width=io.tensor_bytes(key)//4//b
    buffers=[io.zeros(s['shape']) for s in c['steps'][0]['inputs']]
    program=CUDAMTPProgram(c['checkpoint'],c['config'],b,c['capacity'],t)
    try:
        for graph_mode in [False,True,True]:
            io.zero(key);io.zero(value);program.graph.enable_cuda_graph(graph_mode)
            for step_index,step in enumerate(c['steps']):
                host=[read(s) for s in step['inputs']]
                for buf,values in zip(buffers,host):io.update(buf,values)
                before=[cpu(key),cpu(value)]
                packed=program.run([buffers[0],buffers[1],key,value,buffers[2],buffers[3],buffers[4]])
                io.slice_into(packed,0,hidden);io.slice_into(packed,t*h,key);io.slice_into(packed,t*h+state_width,value)
                for output_index,(actual,spec) in enumerate(zip([hidden,key,value],step['expected'])):
                    overall=max(overall,compare(cpu(actual),read(spec),(case_index,graph_mode,step_index,output_index)))
                assert all(cpu(buf)==values for buf,values in zip(buffers,host)),'MTP input mutated'
                # Verify bitwise preservation of every cache slot not written by this chunk.
                for cache,old in zip([key,value],before):
                    current=cpu(cache);dim=state_shape[3];capacity=state_shape[2];kv=state_shape[1]
                    for row in range(b):
                        written={int(host[3][row*t+j]) for j in range(t) if host[4][row*t+j]}
                        for head in range(kv):
                            for slot in range(capacity):
                                if slot in written:continue
                                start=((row*kv+head)*capacity+slot)*dim
                                assert struct.pack('<'+str(dim)+'f',*current[start:start+dim])==struct.pack('<'+str(dim)+'f',*old[start:start+dim]),'Unwritten KV changed'
        stats=json.loads(program.graph.stats_json())
        assert stats['graph_captures']>0 and stats['graph_replays']>=2 and stats['graph_capture_failures']==0,stats
        program.close();embedding=io.embedding([0]*b);assert len(cpu(embedding))==b*h,'Program close broke borrowed IO'
        print('QWEN35_MTP_COMPONENT_CASE_PASS' if len(failures)==initial_failures else 'QWEN35_MTP_COMPONENT_CASE_NUMERICAL_FAILURE',case_index,'batch',b,'chunk',t,'max_abs',overall,flush=True)
    finally:program.close();io.close()
if collect:
    with open(sys.argv[1]+'.failures.json','w') as stream:json.dump(failures,stream,indent=2)
print('QWEN35_MTP_COMPONENT_COMPLETE','cases',len(cases),'max_abs',overall,'numerical_failures',len(failures),flush=True)
assert not failures,'Strict MTP numerical checks failed; inspect saved diagnostics'
