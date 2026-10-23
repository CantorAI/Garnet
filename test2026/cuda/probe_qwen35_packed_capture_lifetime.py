"""Isolate target capture from MTP/replay; sanitizer diagnostic, not a benchmark."""
import hashlib,json,os,struct,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet
from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder
fixture=json.load(open(sys.argv[1]));output=sys.argv[2]
graph=sys.argv[3]=='1';serial=sys.argv[4]=='1';events=[]
def emit(stage,**values):
    events.append({'stage':stage,**values})
    with open(output+'.progress.json','w') as stream:json.dump(events,stream,indent=2)
emit('construct_begin',cuda_graph=graph,serial_target=serial,
    static_binding_reuse=os.environ.get('GARNET_TRT_REUSE_STATIC_BINDINGS'),
    sync_temporary_allocator=os.environ.get('GARNET_TRT_SYNC_TEMPORARY_ALLOCATOR'))
model=CUDAPackedDecoder(fixture['checkpoint'],fixture['config'],8,128,64,4,retain_linear_inputs=True,register_delta_128=True,flat_concat=True)
model.resident_controls=True;model.serial_target_graph_reuse=serial;model.enable_cuda_graph(graph)
emit('constructed')
try:
    ids=fixture['steps'][0]['tokens'];tokens=[[ids[row%len(ids)] for row in range(8)] for _ in range(67)]
    embeddings=[model.io.embedding(row) for row in tokens]
    positions=[[[t]*3 for t in range(67)] for _ in range(8)];reference=None;trials=[]
    for trial in range(3):
        model.reset();emit('prefill_begin',trial=trial)
        predicted=list(model.prefill_sequence(embeddings,positions,[67]*8));emit('prefill_complete',trial=trial)
        for step in range(3):
            emit('step_begin',trial=trial,step=step);predicted=list(model.step(predicted,[1]*8));emit('step_complete',trial=trial,step=step)
        hashes=[]
        for value in (model.last_hidden,model.state):
            values=garnet.tensor_to_cpu(value).tolist();hashes.append(hashlib.sha256(struct.pack('<'+str(len(values))+'f',*values)).hexdigest())
        result={'tokens':predicted,'slots':list(model.slots),'hashes':hashes}
        if reference is None:reference=result
        else:assert result==reference,'Repeated target trial changed states/tokens'
        trials.append(result);emit('trial_complete',trial=trial)
    with open(output,'w') as stream:json.dump({'cuda_graph':graph,'serial_target_graph_reuse':serial,'all_repeats_equal':True,'trials':trials,'memory':model.memory_info(),
        'scope':'Two-layer target-only lifetime isolation; no MTP, replay, pretrained quality or performance acceptance'},stream,indent=2)
    emit('complete')
finally:model.close()
