import sys,json,math
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen35.vl_0_8b.prefill_layer import T,PrefillLayer
with open(sys.argv[1]) as stream:f=json.load(stream)
c=f['config'];batch=len(f['steps'][0]['inputs'][0])//c['hidden_size'];chunk=4;d=c['hidden_size']
kind='linear_attention' if c['kind']=='linear' else 'full_attention';capacity=c.get('capacity',8)
first=f['steps'][0]['inputs']
if c['kind']=='linear':
    channels=2*c['linear_num_key_heads']*c['linear_key_head_dim']+c['linear_num_value_heads']*c['linear_value_head_dim']
    states=[[batch,channels,c['linear_conv_kernel_dim']],[batch,c['linear_num_value_heads'],c['linear_key_head_dim'],c['linear_value_head_dim']]]
else:states=[[batch,c['num_key_value_heads'],capacity,c['head_dim']]]*2
shapes=[[batch,chunk,d],*states,[batch,chunk,3],[batch,chunk],[batch,chunk]]
symbols=[tensor.input('chunk_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
@T.fusion(name='test_prefill',boundary='required')
def forward(*args):return PrefillLayer(*args,c,c['layer_idx'],batch,chunk,capacity,kind)
graph=garnet.compiled_graph()
try:
    graph.build(forward(*symbols),symbols,f['checkpoint'])
    state1=tensor.tensor(first[1],shape=states[0],dtype=tensor.float32)
    state2=tensor.tensor(first[2],shape=states[1],dtype=tensor.float32)
    io=garnet.tensor_io()
    for start in range(0,len(f['steps']),chunk):
        steps=f['steps'][start:start+chunk];hidden=[];positions=[];slots=[];active=[];expected=[]
        state_width=len(steps[-1]['expected'])//batch-d
        for b in range(batch):
            for step in steps:
                hidden+=step['inputs'][0][b*d:(b+1)*d]
                active.append(step['inputs'][-1][b])
                positions+=step['inputs'][3][b*3:b*3+3] if c['kind']=='full' else [0,0,0]
                slots.append(step['inputs'][4][b] if c['kind']=='full' else 0)
                expected+=step['expected'][b*(d+state_width):b*(d+state_width)+d]
            for pad in range(chunk-len(steps)):
                hidden+=[0]*d;active.append(0);positions+=[0,0,0];slots.append(0);expected+=[0]*d
            expected+=steps[-1]['expected'][b*(d+state_width)+d:(b+1)*(d+state_width)]
        values=[tensor.tensor(hidden,shape=shapes[0],dtype=tensor.float32),state1,state2,
            tensor.tensor(positions,shape=shapes[3],dtype=tensor.float32),tensor.tensor(slots,shape=shapes[4],dtype=tensor.float32),
            tensor.tensor(active,shape=shapes[5],dtype=tensor.float32)]
        packed=graph.run(values)
        actual_values=packed.tolist();assert len(actual_values)==len(expected)
        for actual,wanted in zip(actual_values,expected):
            assert math.isfinite(actual) and abs(actual-wanted)<=5e-6+5e-5*abs(wanted)
        width=1
        for value in states[0][1:]:width*=value
        state1=io.slice(packed,chunk*d,states[0]);state2=io.slice(packed,chunk*d+width,states[1])
    print('causal prefill chunk',chunk,kind,'batch',batch,'reference PASS',flush=True)
finally:graph.close();garnet.shutdown()
