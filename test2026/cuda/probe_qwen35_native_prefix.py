"""Production native-prefix requests versus a separately loaded ordinary target.

Hashes stream small column chunks to bound host/device diagnostic storage.
Each image is freshly prepared. Hashing invalidates performance measurements.
"""
import array,hashlib,json,math,os,sys,time
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import garnet
from runtime.cuda_serving import Model
from qwen35_benchmark_workload import batch_inputs,input_hashes
checkpoint,source,output=sys.argv[1:4]
index=int(sys.argv[4]) if len(sys.argv)>4 else 8
shared_output=len(sys.argv)>5 and sys.argv[5]=='1'
whole_program=len(sys.argv)>6 and sys.argv[6]=='1'
finite_bytes=len(sys.argv)>7 and sys.argv[7]=='1'
pack_prompt=len(sys.argv)>8 and sys.argv[8]=='1'
reuse_head=len(sys.argv)>9 and sys.argv[9]=='1'
join_decode_inputs=len(sys.argv)>10 and sys.argv[10]=='1'
record=json.loads(Path(source).read_text())[index];b=record['batch']
_,images=batch_inputs(record['dataset'],record['trial'],record['workload_group'],b,True)
assert input_hashes(images)==record['input_sha256']
prompt='Describe the main objects in this image in one sentence.'
profile=dict(record['profile']);profile.update({'deferPackedStateCommit':True,'nativePackedPrefixCommit':True,
    'serialTargetGraphReuse':True,'retainLinearInputs':True,'sharePackedOutputStorage':shared_output})
if pack_prompt:profile['packPrompt']=True
if reuse_head:profile['reuseVerifiedHead']=True
if join_decode_inputs:profile['joinDecodeInputProjections']=True
assert profile['speculativeMTP']==3 and profile['packedDecoderProgram'] and profile['registerDelta128']

def hashes(model):
    result={};d=model.decoder
    for name,value,width in (('last_hidden',d.last_hidden,d.config['hidden_size']),
                              ('canonical_state',d.state,d.programs[d.decode_chunk_size].state_width)):
        h=hashlib.sha256();words=0
        for start in range(0,width,131072):
            n=min(131072,width-start)
            chunk=d.io.slice(value,start,[b,n])
            if finite_bytes:
                data=d.io.finite_f32_bytes(chunk);assert len(data)==4*b*n
                h.update(data);words+=b*n;del chunk,data
            else:
                host=garnet.tensor_to_cpu(chunk);values=host.tolist()
                assert len(values)==b*n and all(math.isfinite(v) for v in values),'Nonfinite target state'
                data=array.array('f',values);assert data.itemsize==4
                if sys.byteorder!='little':data.byteswap()
                h.update(data.tobytes());words+=len(values)
                del chunk,host,values,data
        result[name]={'sha256':h.hexdigest(),'fp32_words':words,'shape':[b,width],
            'hash_order':'Column chunks <=131072, each chunk in row-major order; all FP32 words finite'}
    return result

trials=[];state_records=[]
print('NATIVE_PREFIX_LOAD_BEGIN',time.time(),b,flush=True)
model=Model(checkpoint,profile,record['model'])
try:
    print('NATIVE_PREFIX_LOAD_COMPLETE',time.time(),flush=True)
    if join_decode_inputs:
        expected=model.config['text_config']['layer_types'].count('linear_attention');native=model.decoder.memory_info()
        assert native['joined_projection_group_count']==expected
        assert sum(g['joined_projection_inputs'] for g in native['graphs'])==expected
    actual_whole=all(g.get('qwen35_whole_program_cuda',False) for g in model.decoder.memory_info()['graphs'])
    assert actual_whole==whole_program,'Pretrained build-policy label differs from actual DLL'
    for trial in range(2):
        r=model.infer_batch([prompt]*b,images,record['request_output_limit'])
        assert r['prompt_token_ids']==record['prompt_token_ids'] and r['speculative']['enabled']
        assert r['decoder_profiles']['native_prefix_commit']
        if reuse_head:
            assert r['speculative']['reuse_verified_head']
            assert r['speculative_metrics']['target_head_evaluations']==r['speculative_metrics']['target_verification_calls']
            assert r['speculative_metrics']['target_head_reuses']==r['speculative_metrics']['target_prefix_replays']>0
        print('NATIVE_PREFIX_INFER_COMPLETE_HASH_BEGIN',trial,time.time(),flush=True)
        trials.append(r);state_records.append(hashes(model))
        print('NATIVE_PREFIX_PRETRAINED_REQUEST',trial,'batch',b,flush=True)
    memory=model.memory_info()
finally:model.close()
# A new engine/model runs only after the speculative model has closed.
ordinary_profile=dict(profile);ordinary_profile.update({'speculativeMTP':0,'speculativeHeadCandidates':0,
    'speculativePrefixReplay':False,'deferPackedStateCommit':False,'nativePackedPrefixCommit':False,'sharePackedOutputStorage':False})
if reuse_head:ordinary_profile['reuseVerifiedHead']=False
print('NATIVE_PREFIX_ORDINARY_LOAD_BEGIN',time.time(),flush=True)
model=Model(checkpoint,ordinary_profile,record['model'])
try:
    print('NATIVE_PREFIX_ORDINARY_LOAD_COMPLETE',time.time(),flush=True)
    ordinary=model.infer_batch([prompt]*b,images,record['request_output_limit'])
    assert not ordinary['speculative']['enabled'],'Ordinary control accidentally used MTP'
    ordinary_states=hashes(model)
finally:model.close()
keys=('generated_token_ids','sampled_tokens_including_stop','finish_reasons','stop_token_ids')
sign=lambda r:{key:r[key] for key in keys}
result={'source_cohort':source,'case_index':index,'batch':b,'input_sha256':record['input_sha256'],
    'reuse_static_bindings_requested':os.environ.get('GARNET_TRT_REUSE_STATIC_BINDINGS')=='1',
    'whole_program_cuda_requested':whole_program,
    'state_hash_method':'bounded_native_finite_bytes' if finite_bytes else 'legacy_finite_tolist_array',
    'native_profile':profile,'ordinary_profile':ordinary_profile,'trials':trials,'ordinary':ordinary,
    'state_hashes':state_records,'ordinary_state_hashes':ordinary_states,'native_memory':memory,
    'same_recorded_sequences':all(r['generated_token_ids']==record['generated_token_ids'] for r in trials),
    'same_ordinary_outputs':all(sign(r)==sign(ordinary) for r in trials),
    'all_state_hashes_equal':all(s==ordinary_states for s in state_records),
    'scope':'Two real-image native-prefix requests, then a separately loaded non-MTP target. Full finite state hash coverage; not an independent math/vision oracle, timing cohort or peak memory.'}
Path(output).write_text(json.dumps(result,indent=2))
print('NATIVE_PREFIX_PRETRAINED_COMPLETE',result['same_recorded_sequences'],result['same_ordinary_outputs'],result['all_state_hashes_equal'],flush=True)
assert result['same_recorded_sequences'] and result['same_ordinary_outputs'] and result['all_state_hashes_equal']
