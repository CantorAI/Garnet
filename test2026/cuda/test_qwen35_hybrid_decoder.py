"""Unwired hybrid state/ownership checks versus one pooled decode profile.

Control closes before the hybrid opens. No engine-comparison timing is taken.
The tiny independent fixture's BF16 dot products avoid large-model rounding
ambiguities; this native equivalence does not close pretrained quality gates.
"""
import hashlib,json,os,struct,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet
from xModel.qwen35.vl_0_8b.cuda_decoder import CUDADecoder
from xModel.qwen35.vl_0_8b.cuda_hybrid import CUDAHybridDecoder
fixture=json.load(open(sys.argv[1]));output=sys.argv[2];b=int(sys.argv[3]) if len(sys.argv)>3 else 1
wide=int(sys.argv[4]) if len(sys.argv)>4 else 64;chunk=4;capacity=256;expected=None;reports=[]
def bits(value):
    data=garnet.tensor_to_cpu(value).tolist();return struct.pack('<'+str(len(data))+'f',*data)
def capture(model):return [bits(model.last_hidden)]+[bits(layer[key]) for layer in model.layers for key in ('state1','state2')]
for hybrid in (False,True):
    if hybrid:model=CUDAHybridDecoder(fixture['checkpoint'],fixture['config'],b,capacity,wide,chunk,retain_linear_inputs=True)
    else:model=CUDADecoder(fixture['checkpoint'],fixture['config'],b,capacity,chunk,True,True,retain_linear_inputs=True,external_weights=True)
    try:
        model.enable_cuda_graph(True);count=67;tokens=[]
        for t in range(count):
            source=fixture['steps'][t%len(fixture['steps'])]['tokens'];tokens.append([source[row%len(source)] for row in range(b)])
        embeddings=[model.io.embedding(row) for row in tokens];lengths=[count-row%4 for row in range(b)]
        positions=[[[t]*3 for t in range(count)] for _ in range(b)];records=[]
        for trial in range(2):
            model.reset();prediction=list(model.prefill_sequence(embeddings,positions,lengths))
            records.append((prediction,list(model.slots),capture(model)))
            for step in range(4):
                active=[int(step!=2 or row%2==0) for row in range(b)]
                prediction=list(model.step(prediction,active))
                records.append((prediction,list(model.slots),capture(model)))
        if expected is None:expected=records
        else:
            if records!=expected:
                differences=[]
                for sample,(reference,actual) in enumerate(zip(expected,records)):
                    if reference[:2]!=actual[:2]:differences.append({'sample':sample,'reference_ids_slots':reference[:2],'actual_ids_slots':actual[:2]})
                    for field,(a,z) in enumerate(zip(reference[2],actual[2])):
                        if a==z:continue
                        count=0;maximum=0;first=[]
                        for index,(av,zv) in enumerate(zip(struct.iter_unpack('<f',a),struct.iter_unpack('<f',z))):
                            if av!=zv:
                                count+=1;maximum=max(maximum,abs(av[0]-zv[0]))
                                if len(first)<3:first.append([index,av[0],zv[0]])
                        differences.append({'sample':sample,'field':field,'different_float_values':count,'max_abs':maximum,'first':first,
                            'reference_sha256':hashlib.sha256(a).hexdigest(),'actual_sha256':hashlib.sha256(z).hexdigest()})
                with open(output,'w') as stream:json.dump({'batch':b,'prefill_chunk':wide,'decode_chunk':chunk,'all_passed':False,'differences':differences},stream,indent=2)
                print('HYBRID_DECODER_DIFFERENCES',json.dumps(differences[:8]),flush=True)
            assert records==expected,'Hybrid prefill/decode differs in IDs/slots/target states'
        memory=model.memory_info();pool=json.loads(model.io.memory_json())
        multiplier=2 if hybrid else 1
        assert sum(g['external_bf16_weight_bytes'] for g in memory['graphs'])==multiplier*pool['projection_weight_pool_bytes']>0
        assert sum(g['external_bf16_weight_inputs'] for g in memory['graphs'])==multiplier*pool['projection_weight_pool_entries']
        assert all(g['graph_captures']>0 and g['graph_replays']>0 and g['graph_capture_failures']==0 for g in memory['graphs'])
        if hybrid:
            assert model.decode.last_hidden is model.wide.last_hidden
            for a,z in zip(model.decode.layers,model.wide.layers):assert a['state1'] is z['state1'] and a['state2'] is z['state2']
            model.reset();full=[embeddings[0]]*capacity;pos=[[[t]*3 for t in range(capacity)] for _ in range(b)]
            model.prefill_sequence(full,pos,[capacity]*b);before=capture(model);rejected=False
            try:model.step([0]*b)
            except BaseException:rejected=True
            assert rejected and model.slots==[capacity]*b and capture(model)==before and not model.poisoned
            model.reset();assert model.slots==[0]*b and model.wide.slots==[0]*b
        reports.append({'hybrid':hybrid,'native_states_ids_slots_equal':True,'memory':memory,'io':pool})
    finally:model.close()
    assert json.loads(model.io.memory_json())['projection_weight_pool_bytes']==0
with open(output,'w') as stream:json.dump({'batch':b,'prefill_chunk':wide,'decode_chunk':chunk,'reports':reports,'all_passed':True,
    'scope':'Synthetic native sequential equivalence and exact shared ownership; no independent quality/pretrained/speed acceptance'},stream,indent=2)
print('QWEN35_HYBRID_DECODER_COMPLETE',b,wide,flush=True)
