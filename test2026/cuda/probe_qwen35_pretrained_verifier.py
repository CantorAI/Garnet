"""Teacher-forced pretrained target arrangement/state equivalence, no speed claim."""
import hashlib,json,os,struct,sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import garnet
from runtime.cuda_serving import Model
from xModel.qwen35.vl_0_8b.cuda_verify import CUDATargetVerifier
from qwen35_benchmark_workload import batch_inputs,input_hashes
checkpoint,cohort,output=sys.argv[1:4];index=int(sys.argv[4]) if len(sys.argv)>4 else 8
record=json.load(open(cohort))[index];assert record['batch']==1 and record['matched_batch_inputs']
names,images=batch_inputs(record['dataset'],record['trial'],record['workload_group'],1,True)
assert input_hashes(images)==record['input_sha256']
teacher=list(record['generated_token_ids'][0])
if record['stop_token_ids'][0] is not None:teacher.append(record['stop_token_ids'][0])
model=Model(checkpoint,record['profile'],record['model']);decoder=model.decoder
verifier=CUDATargetVerifier(decoder,checkpoint,min(4,decoder.chunk_size))
def cpu_bytes(t):
    values=garnet.tensor_to_cpu(t).tolist();return struct.pack('<'+str(len(values))+'f',*values)
state_tensors=[('last_hidden',decoder.last_hidden)]
for layer_index,layer in enumerate(decoder.layers):
    for key in ('state1','state2'):state_tensors.append(('layer_'+str(layer_index)+'_'+key,layer[key]))
try:
    decoder.reset();prepared=model._prepare('Describe the main objects in this image in one sentence.',images[0])
    assert prepared[2]==record['prompt_token_ids'][0]
    first=list(decoder.prefill_sequence(prepared[0],[prepared[1]],[len(prepared[0])]))[0]
    verified=[first];counts=[]
    for start in range(0,len(teacher)-1,verifier.max_tokens):
        n=min(verifier.max_tokens,len(teacher)-1-start)
        tokens=[[teacher[start+t]] for t in range(n)]
        positions=[[[prepared[3]+start+t]*3 for t in range(n)]]
        result=verifier.verify(tokens,[[1]]*n,positions)
        verified.extend(result[0][:n]);verifier.commit_prefix([n]);counts.append(n)
    multi_slots=list(decoder.slots);multi_states={name:cpu_bytes(t) for name,t in state_tensors}
    decoder.reset();first=list(decoder.prefill_sequence(prepared[0],[prepared[1]],[len(prepared[0])]))[0]
    serial=[first]
    for step,token in enumerate(teacher[:-1]):serial.append(list(decoder.step([token],[1],[[prepared[3]+step]*3]))[0])
    states=[]
    for name,tensor in state_tensors:
        actual=cpu_bytes(tensor);expected=multi_states[name];equal=actual==expected
        maximum=0
        if not equal:
            a=struct.unpack('<'+str(len(actual)//4)+'f',actual);e=struct.unpack('<'+str(len(expected)//4)+'f',expected)
            maximum=max(abs(x-y) for x,y in zip(a,e))
        states.append({'name':name,'bitwise_equal':equal,'max_abs':maximum,
            'multi_sha256':hashlib.sha256(expected).hexdigest(),'serial_sha256':hashlib.sha256(actual).hexdigest()})
    result={'source_cohort':cohort,'case_index':index,'profile':record['profile'],'teacher_tokens':teacher,
        'verified_tokens':verified,'serial_tokens':serial,'target_call_chunk_counts':counts,
        'multi_matches_serial_tokens':verified==serial,'serial_matches_recorded_tokens':serial==teacher,
        'multi_matches_recorded_tokens':verified==teacher,'states':states,'all_states_bitwise_equal':all(s['bitwise_equal'] for s in states),
        'multi_slots':multi_slots,'serial_slots':list(decoder.slots),'model_memory':model.memory_info(),'verifier_memory':verifier.memory_info(),
        'scope':'Teacher-forced target correctness diagnostic; native prepared image reused for serial control. No performance, independent vision/quality, speculative serving or target-cache acceptance claim.'}
    with open(output,'w') as stream:json.dump(result,stream,indent=2)
    print('QWEN35_PRETRAINED_VERIFIER_COMPLETE','tokens_equal',result['multi_matches_serial_tokens'],
        'states_equal',result['all_states_bitwise_equal'],'recorded_equal',result['serial_matches_recorded_tokens'],flush=True)
    assert verified==serial==teacher and result['all_states_bitwise_equal'] and multi_slots==list(decoder.slots),'Pretrained verification equivalence failed'
finally:verifier.close();model.close()
