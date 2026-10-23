"""Compare state sharing to independent explicit copies at identical shapes.

The control owns separate constant-weight decoder states for prefill/decode,
copies all state and last-hidden at the boundary, then decodes normally.
It closes before the pooled candidate loads. Cross-shape arithmetic differences
from test_qwen35_hybrid_decoder remain a separate, unchanged failing diagnostic.
"""
import json,os,struct,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet
from xModel.qwen35.vl_0_8b.cuda_decoder import CUDADecoder
from xModel.qwen35.vl_0_8b.cuda_hybrid import CUDAHybridDecoder
fixture=json.load(open(sys.argv[1]));output=sys.argv[2];b=int(sys.argv[3]);wide=int(sys.argv[4]);capacity=256;chunk=4
class ExplicitCopyControl:
    def __init__(self):
        self.decode=CUDADecoder(fixture['checkpoint'],fixture['config'],b,capacity,chunk,True,True,retain_linear_inputs=True)
        try:self.prefill=CUDADecoder(fixture['checkpoint'],fixture['config'],b,capacity,wide,True,True)
        except:self.decode.close();raise
    def __getattr__(self,name):return getattr(self.decode,name)
    def reset(self):self.prefill.reset();self.decode.reset()
    def enable_cuda_graph(self,enabled):self.prefill.enable_cuda_graph(enabled);self.decode.enable_cuda_graph(enabled)
    def prefill_sequence(self,hidden,positions,lengths):
        prediction=self.prefill.prefill_sequence(hidden,positions,lengths)
        for source,destination in zip(self.prefill.layers,self.decode.layers):
            for key in ('state1','state2'):self.decode.io.copy_into(source[key],destination[key])
        self.decode.io.copy_into(self.prefill.last_hidden,self.decode.last_hidden)
        self.decode.slots=list(self.prefill.slots);return prediction
    def memory_info(self):return {'prefill':self.prefill.memory_info(),'decode':self.decode.memory_info(),'scope':'Two independent state/table/constant-weight sets, synthetic explicit-copy control only; no performance comparison'}
    def close(self):
        try:self.prefill.close()
        finally:self.decode.close()
def bits(value):
    data=garnet.tensor_to_cpu(value).tolist();return struct.pack('<'+str(len(data))+'f',*data)
def states(model):return [bits(model.last_hidden)]+[bits(layer[key]) for layer in model.layers for key in ('state1','state2')]
reference=None;reports=[]
for pooled in (False,True):
    model=CUDAHybridDecoder(fixture['checkpoint'],fixture['config'],b,capacity,wide,chunk,retain_linear_inputs=True) if pooled else ExplicitCopyControl()
    try:
        model.enable_cuda_graph(True);count=67;tokens=[]
        for t in range(count):
            source=fixture['steps'][t%len(fixture['steps'])]['tokens'];tokens.append([source[row%len(source)] for row in range(b)])
        embeddings=[model.io.embedding(row) for row in tokens];lengths=[count-row%4 for row in range(b)]
        positions=[[[t]*3 for t in range(count)] for _ in range(b)];records=[]
        # Wide final norm runs once/request: warm, capture, then replay.
        for trial in range(3):
            model.reset();prediction=list(model.prefill_sequence(embeddings,positions,lengths))
            records.append((prediction,list(model.slots),states(model)))
            for step in range(4):
                active=[int(step!=2 or row%2==0) for row in range(b)]
                prediction=list(model.step(prediction,active));records.append((prediction,list(model.slots),states(model)))
        if reference is None:reference=records
        else:assert records==reference,'Shared state differs from independent explicit-copy control at identical profiles'
        memory=model.memory_info()
        if pooled:
            io=json.loads(model.io.memory_json())
            assert sum(g['external_bf16_weight_bytes'] for g in memory['graphs'])==2*io['projection_weight_pool_bytes']>0
            assert sum(g['external_bf16_weight_inputs'] for g in memory['graphs'])==2*io['projection_weight_pool_entries']
            assert all(g['graph_captures']>0 and g['graph_replays']>0 and not g['graph_capture_failures'] for g in memory['graphs']),[(i,g['graph_captures'],g['graph_replays'],g['graph_capture_failures']) for i,g in enumerate(memory['graphs'])]
            assert model.decode.last_hidden is model.wide.last_hidden
            for a,z in zip(model.decode.layers,model.wide.layers):assert a['state1'] is z['state1'] and a['state2'] is z['state2']
            model.reset();full=[embeddings[0]]*capacity;pos=[[[t]*3 for t in range(capacity)] for _ in range(b)]
            model.prefill_sequence(full,pos,[capacity]*b);before=states(model);rejected=False
            try:model.step([0]*b)
            except BaseException:rejected=True
            assert rejected and model.slots==[capacity]*b and states(model)==before and not model.poisoned
            model.reset();assert model.slots==model.wide.slots==[0]*b
        reports.append({'pooled':pooled,'states_ids_slots_bitwise_equal':True,'memory':memory})
    finally:model.close()
    assert json.loads(model.io.memory_json())['projection_weight_pool_bytes']==0
with open(output,'w') as stream:json.dump({'batch':b,'prefill_chunk':wide,'decode_chunk':chunk,'reports':reports,'all_passed':True,
    'scope':'Native identical-shape explicit-copy versus shared-state correctness. Cross-shape precision, independent quality, pretrained behavior and speed remain separate.'},stream,indent=2)
print('QWEN35_HYBRID_COPY_CONTROL_COMPLETE',b,wide,flush=True)
