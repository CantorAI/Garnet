"""Sequential same-profile whole-decoder versus per-layer hybrid control."""
import json,os,struct,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet
from xModel.qwen35.vl_0_8b.cuda_hybrid import CUDAHybridDecoder
from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder
fixture=json.load(open(sys.argv[1]));output=sys.argv[2];b=int(sys.argv[3]);p=int(sys.argv[4]);d=4;c=256;expected=None;reports=[]
resident=len(sys.argv)>5 and sys.argv[5]=='1';use_packed=not(len(sys.argv)>6 and sys.argv[6]=='0')
register_delta=len(sys.argv)>7 and sys.argv[7]=='1';flat_concat=len(sys.argv)>8 and sys.argv[8]=='1'
serial_target=len(sys.argv)>9 and sys.argv[9]=='1'
def bits(value):
    values=garnet.tensor_to_cpu(value).tolist();return struct.pack('<'+str(len(values))+'f',*values)
def state(model,whole):
    if whole:canonical=bits(model.state)
    else:
        blobs=[bits(layer[key]) for layer in model.layers for key in ('state1','state2')]
        canonical=b''.join(blob[row*(len(blob)//b):(row+1)*(len(blob)//b)] for row in range(b) for blob in blobs)
    return bits(model.last_hidden),canonical
for whole in (False,True):
    factory=CUDAPackedDecoder if whole and use_packed else CUDAHybridDecoder
    extra={'flat_concat':flat_concat} if whole and use_packed else {}
    model=factory(fixture['checkpoint'],fixture['config'],b,c,p,d,retain_linear_inputs=True,register_delta_128=register_delta and whole,**extra)
    model.resident_controls=resident and whole
    model.serial_target_graph_reuse=serial_target and whole
    try:
        model.enable_cuda_graph(True);count=67;tokens=[]
        for t in range(count):
            source=fixture['steps'][t%len(fixture['steps'])]['tokens'];tokens.append([source[row%len(source)] for row in range(b)])
        embeddings=[model.io.embedding(token) for token in tokens];positions=[[[t]*3 for t in range(count)] for _ in range(b)]
        lengths=[count-row%4 for row in range(b)];records=[]
        for trial in range(3):
            model.reset();ids=list(model.prefill_sequence(embeddings,positions,lengths));records.append((ids,list(model.slots),state(model,whole and use_packed)))
            for t in range(5):
                flags=[int(t!=2 or row%2==0) for row in range(b)]
                ids=list(model.step(ids,flags));records.append((ids,list(model.slots),state(model,whole and use_packed)))
        if expected is None:expected=records
        else:assert records==expected,'Packed parent differs from same-shape hybrid control'
        memory=model.memory_info()
        if os.environ.get('GARNET_TRT_ZERO_AUX_STREAMS')=='1':assert all(g['aux_streams']==0 for g in memory['graphs'])
        if whole and serial_target:
            assert all(g['contexts']==1 and g['serial_graph_reuse_requested'] and g['serial_graph_reuses']>0 for g in memory['graphs'])
        assert all(g['graph_captures'] and g['graph_replays'] and not g['graph_capture_failures'] for g in memory['graphs'])
        if whole:
            pool=json.loads(model.io.memory_json())
            assert sum(g['external_bf16_weight_bytes'] for g in memory['graphs'])==2*pool['projection_weight_pool_bytes']>0
            model.reset();model.prefill_sequence([embeddings[0]]*c,[[[t]*3 for t in range(c)] for _ in range(b)],[c]*b)
            before=state(model,whole and use_packed);rejected=False
            try:model.step([0]*b)
            except ValueError:rejected=True
            assert rejected and state(model,whole and use_packed)==before and model.slots==[c]*b and not model.poisoned
            memory=model.memory_info()
            if serial_target:assert all(g['contexts']==1 for g in memory['graphs']),memory
        reports.append({'whole':whole,'all_state_ids_slots_equal':True,'memory':memory})
    finally:model.close()
    assert json.loads(model.io.memory_json())['projection_weight_pool_bytes']==0
with open(output,'w') as stream:json.dump({'all_passed':True,'batch':b,'prefill':p,'decode':d,'resident_controls':resident,'serial_target_graph_reuse':serial_target,'candidate_packed':use_packed,'register_delta_128':register_delta,'flat_concat':flat_concat,'records':18,'reports':reports,
    'scope':'Native two-layer same-shape integration control; no independent pretrained quality/speed acceptance'},stream,indent=2)
print('QWEN35_PACKED_DECODER_COMPLETE',b,p,flush=True)
