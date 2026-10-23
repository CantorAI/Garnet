"""One decoder at a time: constants versus explicit pooled BF16 projections."""
import json,os,struct,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet
from xModel.qwen35.vl_0_8b.cuda_decoder import CUDADecoder
fixture=json.load(open(sys.argv[1]));output=sys.argv[2];chunk=int(sys.argv[3]) if len(sys.argv)>3 else 16
b=fixture['batch'];control=None;reports=[]
def bits(value):
    data=garnet.tensor_to_cpu(value).tolist();return struct.pack('<'+str(len(data))+'f',*data)
for external in (False,True):
    model=CUDADecoder(fixture['checkpoint'],fixture['config'],b,fixture['capacity'],chunk,True,True,external_weights=external)
    try:
        model.enable_cuda_graph(True);records=[]
        for trial in range(2):
            model.reset()
            for step in fixture['steps']:
                prediction=list(model.step(step['tokens'],step['active']))
                tensors=[model.hidden,model.last_hidden]+[layer[key] for layer in model.layers for key in ('state1','state2')]
                records.append((prediction,list(model.slots),[bits(value) for value in tensors]))
        if control is None:control=records
        else:assert records==control,'Pooled projection decoder differs from constant decoder'
        memory=model.memory_info();borrowed=sum(g['external_bf16_weight_bytes'] for g in memory['graphs'])
        if external:assert borrowed==memory['projection_weight_pool_bytes']>0
        else:assert borrowed==memory['projection_weight_pool_bytes']==0
        reports.append({'external':external,'memory':memory,'all_native_states_predictions_equal':True})
    finally:model.close()
with open(output,'w') as stream:json.dump({'batch':b,'chunk':chunk,'reports':reports,'all_passed':True,
    'scope':'Sequential native constant-versus-pooled equivalence, not independent math or performance acceptance'},stream,indent=2)
print('QWEN35_POOLED_DECODER_COMPLETE',b,chunk,flush=True)
