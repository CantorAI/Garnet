"""D8/MTP7 real-image state preservation against a separate ordinary target.

Streams finite FP32 hashes in bounded chunks. This is a correctness probe;
hashing and host transfers disqualify these runs as performance evidence.
"""
import hashlib,json,os,sys,time
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import garnet
from runtime.cuda_serving import Model
from qwen35_benchmark_workload import batch_inputs,input_hashes
checkpoint,source,output=sys.argv[1:4]
events=[]
def progress(stage,trial=None):
    events.append({'stage':stage,'trial':trial,'unix_seconds':time.time()})
    Path(output+'.progress.json').write_text(json.dumps(events,indent=2))
    print('WIDE8_PROBE_PROGRESS',stage,trial,flush=True)
progress('read_source_begin')
whole_program=len(sys.argv)>4 and sys.argv[4]=='1'
record=json.loads(Path(source).read_text())[8];b=record['batch']
_,images=batch_inputs(record['dataset'],record['trial'],record['workload_group'],b,True)
assert input_hashes(images)==record['input_sha256']
prompt='Describe the main objects in this image in one sentence.'
profile=dict(record['profile']);profile.update({'decodeChunk':8,'speculativeMTP':7,
    'deferPackedStateCommit':True,'nativePackedPrefixCommit':True,'serialTargetGraphReuse':True,
    'retainLinearInputs':True,'sharePackedOutputStorage':True})
assert profile['packedDecoderProgram'] and profile['registerDelta128']

def hashes(model):
    result={};d=model.decoder
    for name,value,width in (('last_hidden',d.last_hidden,d.config['hidden_size']),
                            ('canonical_state',d.state,d.programs[d.decode_chunk_size].state_width)):
        h=hashlib.sha256();words=0
        for start in range(0,width,131072):
            n=min(131072,width-start)
            chunk=d.io.slice(value,start,[b,n]);data=d.io.finite_f32_bytes(chunk)
            assert len(data)==4*b*n
            h.update(data);words+=b*n;del chunk,data
        result[name]={'sha256':h.hexdigest(),'fp32_words':words,'shape':[b,width],
            'hash_order':'Column chunks <=131072, each chunk row-major, finite FP32 words'}
    return result

trials=[];states=[]
print('WIDE8_PRETRAINED_LOAD_BEGIN',time.time(),b,flush=True)
progress('speculative_load_begin')
model=Model(checkpoint,profile,record['model'])
try:
    progress('speculative_load_complete')
    print('WIDE8_PRETRAINED_LOAD_COMPLETE',time.time(),flush=True)
    actual_whole=all(g.get('qwen35_whole_program_cuda',False) for g in model.decoder.memory_info()['graphs'])
    assert actual_whole==whole_program,'Wide8 pretrained build policy differs from DLL'
    for trial in range(2):
        progress('speculative_infer_begin',trial)
        r=model.infer_batch([prompt]*b,images,record['request_output_limit'])
        assert r['prompt_token_ids']==record['prompt_token_ids'] and r['speculative']['mtp_depth']==7
        assert r['decoder_profiles']['decode_chunk']==8 and r['decoder_profiles']['native_prefix_commit']
        print('WIDE8_PRETRAINED_INFER_COMPLETE_HASH_BEGIN',trial,time.time(),flush=True)
        progress('speculative_hash_begin',trial)
        trials.append(r);states.append(hashes(model))
        progress('speculative_hash_complete',trial)
        print('WIDE8_PRETRAINED_REQUEST',trial,'batch',b,flush=True)
finally:model.close()
ordinary_profile=dict(profile);ordinary_profile.update({'speculativeMTP':0,'speculativeHeadCandidates':0,
    'speculativePrefixReplay':False,'deferPackedStateCommit':False,'nativePackedPrefixCommit':False,
    'sharePackedOutputStorage':False})
print('WIDE8_ORDINARY_LOAD_BEGIN',time.time(),flush=True)
progress('ordinary_load_begin')
model=Model(checkpoint,ordinary_profile,record['model'])
try:
    progress('ordinary_infer_begin')
    print('WIDE8_ORDINARY_LOAD_COMPLETE',time.time(),flush=True)
    ordinary=model.infer_batch([prompt]*b,images,record['request_output_limit'])
    assert not ordinary['speculative']['enabled']
    progress('ordinary_hash_begin')
    ordinary_state=hashes(model)
    progress('ordinary_hash_complete')
finally:model.close()
keys=('generated_token_ids','sampled_tokens_including_stop','finish_reasons','stop_token_ids')
sign=lambda r:{key:r[key] for key in keys}
result={'source_cohort':source,'case_index':8,'batch':b,'input_sha256':record['input_sha256'],
    'state_hash_method':'bounded_native_finite_bytes',
    'whole_program_cuda_requested':whole_program,'native_profile':profile,'ordinary_profile':ordinary_profile,
    'trials':trials,'ordinary':ordinary,'state_hashes':states,'ordinary_state_hashes':ordinary_state,
    'same_saved_d4_sequences':all(r['generated_token_ids']==record['generated_token_ids'] for r in trials),
    'same_ordinary_outputs':all(sign(r)==sign(ordinary) for r in trials),
    'all_state_hashes_equal':all(s==ordinary_state for s in states),
    'scope':'D8/MTP7 repeated real request versus separate ordinary D8 target; full finite state hashes. Not independent quality or latency.'}
Path(output).write_text(json.dumps(result,indent=2))
assert result['same_ordinary_outputs'] and result['all_state_hashes_equal'],'Wide8 state/output preservation failed'
progress('complete')
print('WIDE8_PRETRAINED_COMPLETE',result['same_saved_d4_sequences'],flush=True)
