"""Actual MTP proposals versus serial pretrained outputs/states; diagnostic only."""
import hashlib,json,os,struct,sys,time
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import garnet
from runtime.cuda_serving import Model
from xModel.qwen35.vl_0_8b.cuda_speculative import CUDASpeculativeSession
from qwen35_benchmark_workload import batch_inputs,input_hashes
checkpoint,cohort,output=sys.argv[1:4];depth=int(sys.argv[4]) if len(sys.argv)>4 else 3
index=int(sys.argv[5]) if len(sys.argv)>5 else 8;record=json.load(open(cohort))[index];b=record['batch']
head_candidates=int(sys.argv[6]) if len(sys.argv)>6 else 0
prefix_replay=len(sys.argv)>7 and sys.argv[7]=='1'
wide=int(sys.argv[8]) if len(sys.argv)>8 else 0;decode=int(sys.argv[9]) if len(sys.argv)>9 else 4
packed_program=len(sys.argv)>10 and sys.argv[10]=='1'
names,images=batch_inputs(record['dataset'],record['trial'],record['workload_group'],b,True)
assert input_hashes(images)==record['input_sha256']
profile=dict(record['profile']);profile['retainLinearInputs']=prefix_replay
if wide:profile.update({'prefillChunk':wide,'decodeChunk':decode,'sharedProjectionWeights':True,'fusedDelta':True,'fusedAttention':True})
if packed_program:profile['packedDecoderProgram']=True
if len(sys.argv)>11 and sys.argv[11]=='1':profile['residentControls']=True
if len(sys.argv)>12 and sys.argv[12]=='1':profile['serialGraphReuse']=True
if len(sys.argv)>13 and sys.argv[13]=='1':profile['registerDelta128']=True
if len(sys.argv)>14 and sys.argv[14]=='1':profile['flatPackedConcat']=True
model=Model(checkpoint,profile,record['model']);decoder=model.decoder
session=CUDASpeculativeSession(decoder,checkpoint,model.stops,depth,head_candidates,prefix_replay)
prompt='Describe the main objects in this image in one sentence.'
state_tensors=[('last_hidden',decoder.last_hidden)]
if packed_program:state_tensors.append(('canonical_state',decoder.state))
else:
    for i,layer in enumerate(decoder.layers):
        for key in ('state1','state2'):state_tensors.append(('layer_'+str(i)+'_'+key,layer[key]))
def state_bytes():
    result={}
    for name,tensor in state_tensors:
        values=garnet.tensor_to_cpu(tensor).tolist();result[name]=struct.pack('<'+str(len(values))+'f',*values)
    return result
trials=[];first_states=None
try:
    for trial in range(2):
        started=time.monotonic();session.reset();prepared=[model._prepare(prompt,image) for image in images]
        lengths=[len(r[0]) for r in prepared];count=max(lengths);zero=decoder.io.zeros([1,decoder.config['hidden_size']])
        hidden=[]
        for t in range(count):hidden.append(decoder.io.stack_rows([r[0][t] if t<len(r[0]) else zero for r in prepared]))
        positions=[r[1]+[[0,0,0]]*(count-len(r[1])) for r in prepared]
        assert [r[2] for r in prepared]==record['prompt_token_ids']
        session.prefill(hidden,positions,lengths,[r[3] for r in prepared]);first=time.monotonic()
        result=session.generate(record['request_output_limit']);done=time.monotonic()
        result.update({'trial':trial,'diagnostic_total_seconds':done-started,'diagnostic_ttft_seconds':first-started,
            'diagnostic_decode_seconds':done-first,'slots':list(decoder.slots),'session_memory':session.memory_info(),'base_model_memory':model.memory_info()})
        current_states=state_bytes()
        if first_states is None:first_states=current_states
        else:assert current_states==first_states,'Repeated speculative request changed target states'
        trials.append(result)
        print('PRETRAINED_SPECULATIVE_TRIAL_COMPLETE',trial,'samples',result['sampled_tokens_including_stop'],
            'target_calls',result['speculative_metrics']['target_forward_calls'],flush=True)
    session.reset();serial=model.infer_batch([prompt]*b,images,record['request_output_limit']);serial_states=state_bytes()
    state_results=[]
    for name,actual in serial_states.items():
        expected=first_states[name]
        state_results.append({'name':name,'bitwise_equal':actual==expected,
            'serial_sha256':hashlib.sha256(actual).hexdigest(),'speculative_sha256':hashlib.sha256(expected).hexdigest()})
    keys=['generated_token_ids','sampled_tokens_including_stop','finish_reasons','stop_token_ids']
    signatures=[{k:row[k] for k in keys} for row in trials];reference={k:serial[k] for k in keys}
    result={'source_cohort':cohort,'case_index':index,'base_profile':record['profile'],'actual_model_profile':profile,'experimental_mtp_depth':depth,'head_candidates':head_candidates,'prefix_replay':prefix_replay,
        'trials':trials,'serial':serial,'serial_slots':list(decoder.slots),'state_results':state_results,
        'speculative_matches_serial':all(s==reference for s in signatures),
        'serial_matches_recorded':serial['generated_token_ids']==record['generated_token_ids'],
        'all_target_states_bitwise_equal':all(s['bitwise_equal'] for s in state_results),
        'scope':'Correctness diagnostic with actual native MTP proposals and fresh native image encoding each trial. Two requests are not a matched timing cohort. Restore/recompute partial commit; no 2x or quality acceptance.'}
    with open(output,'w') as stream:json.dump(result,stream,indent=2)
    print('QWEN35_PRETRAINED_SPECULATIVE_COMPLETE','outputs',result['speculative_matches_serial'],
        'states',result['all_target_states_bitwise_equal'],'recorded',result['serial_matches_recorded'],flush=True)
    assert result['speculative_matches_serial'] and result['all_target_states_bitwise_equal'] and result['serial_matches_recorded']
    assert all(r['slots']==list(decoder.slots) for r in trials)
finally:session.close();model.close()
