"""Separate old/joined fixture decoders, full finite states and explicit owners."""
import hashlib,json,os,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder
fixture=json.load(open(sys.argv[1]));output=sys.argv[2];config=fixture['config'];checkpoint=fixture['checkpoint']
records=[]
def digest(model,value):
    # Preserve every finite FP32 word without per-element Python conversion.
    width=model.programs[model.decode_chunk_size].state_width
    assert model.io.tensor_bytes(value)==model.batch*width*4
    step=min(width,1048576//model.batch);result=hashlib.sha256()
    for start in range(0,width,step):
        part=model.io.slice(value,start,[model.batch,min(step,width-start)])
        result.update(model.io.finite_f32_bytes(part))
    return result.hexdigest()
def run(batch,joined):
    print('JOINED_DECODER_CASE_BEGIN',batch,joined,flush=True)
    model=CUDAPackedDecoder(checkpoint,config,batch,32,32,4,register_delta_128=True,flat_concat=True,
        shared_output_storage=True,join_decode_input_projections=joined)
    model.resident_controls=True;model.serial_target_graph_reuse=True;model.enable_cuda_graph(True)
    states=[]
    def snapshot(ids):
        states.append({'predictions':list(ids),'slots':list(model.slots),'state_sha256':digest(model,model.state),
            'hidden_sha256':hashlib.sha256(model.io.finite_f32_bytes(model.last_hidden)).hexdigest()})
    try:
        prompt=[]
        for t in range(5):
            ids=fixture['steps'][t%4]['tokens'];prompt.append(model.io.embedding([ids[row%len(ids)] for row in range(batch)]))
        positions=[[[t]*3 for t in range(5)] for _ in range(batch)];lengths=[5-row%3 for row in range(batch)]
        snapshot(model.prefill_sequence(prompt,positions,lengths))
        for step in range(8):
            ids=fixture['steps'][step%4]['tokens'];tokens=[ids[row%len(ids)] for row in range(batch)]
            flags=[int((step+row)%4!=2) for row in range(batch)]
            snapshot(model.step(tokens,flags,[[slot]*3 for slot in model.slots]))
        info=model.memory_info();groups=info['joined_projection_group_count']
        expected=config['layer_types'].count('linear_attention') if joined else 0
        assert groups==expected
        assert sum(g['joined_projection_inputs'] for g in info['graphs'])==expected
        assert all(g['graph_capture_failures']==0 for g in info['graphs'])
        return states,info
    finally:model.close()
for batch in (1,2,4,8):
    baseline,old=run(batch,False)
    candidate,new=run(batch,True)
    with open(output,'w') as stream:json.dump({'records':records,'current_batch':batch,'baseline_snapshots':baseline,
        'candidate_snapshots':candidate,'all_passed':False},stream,indent=2)
    assert baseline==candidate,('Joined full-state/output mismatch',batch,
        [i for i,(a,b) in enumerate(zip(baseline,candidate)) if a!=b])
    assert old['projection_weight_pool_bytes']==new['projection_weight_pool_bytes']
    records.append({'batch':batch,'snapshots':candidate,'same_all_full_finite_states_and_ids':True,
        'old_weight_bytes':old['projection_weight_pool_bytes'],'joined_weight_bytes':new['projection_weight_pool_bytes'],
        'native_joined_inputs':sum(g['joined_projection_inputs'] for g in new['graphs'])})
with open(output,'w') as stream:json.dump({'records':records,'all_passed':True,
    'scope':'Sequential independent baseline/joined fixture models; prefill and eight masked steps; full finite state preservation. Not pretrained quality or speed acceptance.'},stream,indent=2)
print('QWEN35_JOINED_DECODER_PASS',len(records),flush=True)
