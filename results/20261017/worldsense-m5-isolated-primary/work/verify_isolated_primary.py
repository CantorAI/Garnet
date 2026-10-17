"""Independent WS54 archive coverage/output/metric audit; CPU only."""
from pathlib import Path
import csv,hashlib,json,statistics,subprocess,sys
r=Path('/Users/shawn/CantorAI/Garnet')
d=Path(sys.argv[1])if len(sys.argv)>1 else r/'results/20261017/worldsense-m5-isolated-primary'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert d.name=='worldsense-m5-isolated-primary'
assert load(d/'protocol.json')['experiment']=='WS-M5-54'
manifest=load(d/'file-manifest.json')
assert set(manifest)=={str(p.relative_to(d))for p in d.rglob('*')if p.is_file()and p.name!='file-manifest.json'}
assert all(sha(d/rel)==h for rel,h in manifest.items())
reconstruction=load(d/'source-reconstruction.json');closure=load(d/reconstruction['source_manifest']);base=reconstruction['base_commit']
refs=[base+':'+rel for rel in closure]
raw=subprocess.check_output(['git','-C',str(r),'cat-file','--batch'],input=('\n'.join(refs)+'\n').encode())
offset=0
for rel,digest in closure.items():
    end=raw.index(b'\n',offset);header=raw[offset:end].split();offset=end+1
    if header[-1]==b'missing':original=None
    else:
        assert header[1]==b'blob';size=int(header[2]);original=raw[offset:offset+size];offset+=size+1
    overlay=d/'source'/rel
    actual=overlay.read_bytes()if overlay.exists()else original
    assert actual is not None and hashlib.sha256(actual).hexdigest()==digest,rel
assert offset==len(raw)
assert closure==load(d/'correctness-source-manifest.json')
assert sorted(reconstruction['overlay_files'])==sorted(load(d/'implementation-audit.json')['changed_files'])
assert len(reconstruction['overlay_files'])==10 and len(closure)==1718
assert all(load(d/'api-correctness-result.json').values())
assert all(v==0 for v in load(d/'native-results.json').values())
assert all(v['returncode']==0 for v in load(d/'resident-results.json'))
assert all((d/('native-'+mode+'-projection-words.bin')).read_bytes()==(d/'native-off-projection-words.bin').read_bytes()for mode in ['primary','on','fault'])
assert load(d/'native-word-comparison.json')['records']==186
corpus=load(d/'corpus-GPU-bits.json');t=corpus['totals']
assert t['packed_matrices']==201 and t['raw_small_matrices']==245
assert t['GPU_original_FP32_words_checked']==t['GPU_vector_original_FP32_words_checked']==4200857600
assert t['retained_primary_and_small_raw_bytes']==9850930176
assert t['HOST_raw_FP32_words_checked']==1724383232
assert sum(m['HOST_raw_FP32_words_checked']for m in corpus['matrices'])==1724383232
assert all(m['HOST_raw_FP32_words_checked']==(0 if m['packed']else m['rows']*m['inner'])for m in corpus['matrices'])
assert set(t['GPU_prefill_tile_FP32_words_checked'])=={'32x32','64x32','64x64'}and all(v==4200857600 for v in t['GPU_prefill_tile_FP32_words_checked'].values())
assert all(set(m['GPU_prefill_tile_FP32_words_checked'])=={'32x32','64x32','64x64'}and all(v==(m['rows']*m['inner']if m['packed']else 0)for v in m['GPU_prefill_tile_FP32_words_checked'].values())for m in corpus['matrices'])
assert len(corpus['matrices'])==446 and all(m['owner_accounting_restored']for m in corpus['matrices'])
assert load(d/'correctness-terminal.json')['all_three_actual_model_modes_and_five_readers_exact']
assert load(d/'correctness-terminal.json')['all_original_numerical_state_token_guards_pass']
assert closure['src/backends/metal/primary_block_shader.h']==load(d/'seed-source-manifest.json')['src/backends/metal/primary_block_shader.h']
assert closure['src/backends/metal/primary_block_codec.h']==load(d/'seed-source-manifest.json')['src/backends/metal/primary_block_codec.h']
off,on=(load(d/('api-state-progress-'+mode+'.json'))['records']for mode in ['off','on'])
assert len(off)==len(on)==5
for a,b in zip(off,on):
    assert a['model_id']==b['model_id']and a['states']==b['states']
    assert all(a['call'][k]==b['call'][k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])

# Actual prior-primary model outputs/states and immutable external build profiles.
prior=load(d/'api-state-progress-primary.json')['records'];assert len(prior)==5
for a,b in zip(off,prior):
    assert a['model_id']==b['model_id']and a['states']==b['states']
    assert len(a['states'])==(32 if a['model_id']=='Qwen3.5-4B'else 28)
    assert all(a['call'][k]==b['call'][k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])
    assert 'primary_bf16_blocks'in b['call']['backend']and'primary_vector_prefill'not in b['call']['backend']
for name,info in load(d/'build-profile-identity.json').items():
    raw=subprocess.check_output(['git','-C',str(r),'show',info['immutable_git']]);assert (d/'build-profile'/name).read_bytes()==raw and sha(d/'build-profile'/name)==info['sha256']
proof=load(d/'independent-shader-proof.json');assert proof['CXX_materialized_OFF_primary_fragment_exact']and proof['CXX_materialized_ON_equals_original_plus_new_fragment']
rawheader=subprocess.check_output(['git','-C',str(r),'show',base+':src/backends/metal/primary_block_shader.h']).decode();original=rawheader.split('R"PRIMARY(',1)[1].split(')PRIMARY"',1)[0]
new=(d/'source/src/backends/metal/primary_vector_prefill_shader.h').read_text().split('R"PVEC(',1)[1].split(')PVEC"',1)[0]
assert (d/'materialized-primary-off.metal').read_text()==original and(d/'materialized-primary-on.metal').read_text()==original+new
for oldname,newname,total,c,k,threads in [('primary_bf16_gemm','primary_vector_bf16_gemm','32*32',32,32,128),('primary_gemm_16_64_64','primary_vector_gemm_16_64_64','4096',64,64,256),('primary_gemm_32_64_32','primary_vector_gemm_32_64_32','2048',64,32,256)]:
    at=original.index('kernel void '+oldname+'(');end=original.find('kernel void ',at+1);before=original[at:end if end>=0 else len(original)].rstrip()
    at=new.index('kernel void '+newname+'(');end=new.find('kernel void ',at+1);after=new[at:end].split('struct PrimaryPrefillTileProof',1)[0].rstrip()
    old=f'for(uint i=tid;i<{total};i+={threads}){{ulong k=base+i/{c},column=colBase+i%{c};right[i]=(k<p.inner&&column<p.columns)?as_type<float>(uint(primary_word(payload,metadata,column,k,p.inner))<<16):0.0f;}}'
    replacement=f'primary_prefill_tile_fill<{c},{k},{threads}>(payload,metadata,right,uint(colBase),uint(base),uint(p.columns),uint(p.inner),tid);'
    declarations={
      'primary_bf16_gemm':('threadgroup float left[16*32],right[32*32],result[16*32];','threadgroup float left[16*32],result[16*32];\n    threadgroup float4 rightAligned[256];threadgroup float* right=(threadgroup float*)rightAligned;'),
      'primary_gemm_16_64_64':('threadgroup float left[1024],right[4096],result[1024];','threadgroup float left[1024],result[1024];\nthreadgroup float4 rightAligned[1024];threadgroup float* right=(threadgroup float*)rightAligned;'),
      'primary_gemm_32_64_32':('threadgroup float left[1024],right[2048],result[2048];','threadgroup float left[1024],result[2048];\nthreadgroup float4 rightAligned[512];threadgroup float* right=(threadgroup float*)rightAligned;')}
    previous,aligned=declarations[oldname];assert after.count(aligned)==1;after=after.replace(aligned,previous,1)
    assert after.replace('kernel void '+newname+'(','kernel void '+oldname+'(',1).replace(replacement,old,1)==before
program=load(d/'actual-program-proof.json');assert program['actual_candidate_main_equals_raw_control']and program['actual_candidate_prefill_equals_minimal_preamble_plus_full_primary_and_vector_fragment']
for mode,data in program['programs'].items():
    for kind,info in data.items():
        path=d/('actual-'+mode+'-'+kind+'-program.metal');assert path.stat().st_size==info['bytes']and sha(path)==info['sha256']
main=(d/'actual-off-main-program.metal').read_bytes()
assert main==(d/'actual-on-main-program.metal').read_bytes()==(d/'actual-fault-main-program.metal').read_bytes()
preamble=(d/'source/src/backends/metal/primary_isolation_shader.h').read_text().split('R"IPRIMARY(',1)[1].split(')IPRIMARY"',1)[0]
hstart=rawheader.index('inline ushort4 primary_quad_kword(');hend=rawheader.index('inline PrimaryQuad primary_quad(',hstart)
abi='struct BF16Params {ulong rows,inner,columns,aStride,oStride;};'
bodyheader=subprocess.check_output(['git','-C',str(r),'show',base+':src/backends/metal/body_shape_shader.h']).decode();constants=bodyheader.split('R"BODYSHAPE(',1)[1].split('kernel void ',1)[0]
assert preamble=='#include <metal_stdlib>\nusing namespace metal;\n'+abi+'\n'+constants
assert (d/'actual-on-prefill-program.metal').read_bytes()==preamble.encode()+original.encode()+new.encode()==(d/'actual-fault-prefill-program.metal').read_bytes()
assert b'kernel void primary_bf16_decode'in(d/'actual-on-prefill-program.metal').read_bytes()
assert not(d/'actual-off-prefill-program.metal').read_bytes()and not(d/'actual-primary-prefill-program.metal').read_bytes()
assert b'kernel void primary_vector_bf16_gemm'not in main
mm=(d/'source/src/backends/metal/metal_context.mm').read_text()
assert 'impl_->kernels->library=library;'in mm and'prefillLibrary=isolated;'in mm
for name in ['primary_vector_bf16_gemm','primary_vector_gemm_16_64_64','primary_vector_gemm_32_64_32','primary_vector_prefill_expand']:
    assert '[prefillLibrary newFunctionWithName:@"'+name+'"]'in mm
assert '[primaryLibrary newFunctionWithName:@"primary_bf16_decode"]'in mm
for mode in ['on','fault']:
    assert 'Vector prefill tile all3 geometries/full finite/mixed words/nonzero rows/tail260/closed-batch/cap/recovery PASS'in(d/(mode+'-LastTest.log')).read_text()

# New K32-only host policy and measured CPU loop proof.
proofCPU=load(d/'cpu-loop-proof.json');assert proofCPU['rowBytes_call_outside_all_backward_branch_intervals']and proofCPU['rowBytes_calls_in_function']==1
assert '#if defined(GARNET_METAL_PRIMARY_VECTOR_PREFILL) && !defined(GARNET_METAL_PRIMARY_VECTOR_PREFILL_K32_ONLY)'in mm
assert 'primaryGemm16Wide=[impl_->device newComputePipelineStateWithFunction:[primaryLibrary newFunctionWithName:@"primary_gemm_16_64_64"]'in mm
assert load(d/'wide-scope-proof.json')['wide_admission_scope_verified']
assert load(d/'wide-scope-proof.json')['full_context_recovered_by_only_library_composition_and_lookup_inverse']
for mode in ['on','fault']:
 assert 'Matrix metadata/live-MPS parity empty/copy/move/reassign/padding/pending/Submit/Abort/host-zero/context-lifetime PASS'in(d/(mode+'-LastTest.log')).read_text()
flags=load(d/'build-flag-comparison.json');assert flags['on']['GARNET_METAL_PRIMARY_VECTOR_PREFILL_K32_ONLY']=='ON'and flags['primary']['GARNET_METAL_PRIMARY_VECTOR_PREFILL_K32_ONLY']=='OFF'
assert load(d/'independent-direct-upload-proof.json')['raw_and_storage_control_programs_byte_exact_WS53']
for mode in ['off','primary','on']:
 assert flags[mode]['GARNET_METAL_CACHED_MATRIX_METADATA']==('ON'if mode=='on'else'OFF')
 assert all(flags[mode][cap]==('OFF'if mode=='off'else'ON')for cap in ['GARNET_METAL_LAZY_RESIDENT_DEFAULTS','GARNET_METAL_WIDE_PRIMARY_STORAGE'])

# New direct staging path: full learned word coverage and immutable fallback.
direct=load(d/'direct-upload-corpus.json');selection=load(d/'direct-upload-selection.json');assert direct['totals']=={'weights':546,'FP32_words_checked':334609152}
expected={(x['model'],x['tensor']):x for x in selection['weights']};actual={(x['model'],x['tensor']):x for x in direct['weights']};assert len(expected)==len(actual)==546 and expected.keys()==actual.keys()
for key,x in expected.items():
 y=actual[key];assert all(x[k]==y[k]for k in ['shape','rows','columns'])and y['FP32_words_checked']==x['rows']*x['columns']and y['storage']=='matrix_fp32'and len(y['BF16_sha256'])==64
assert load(d/'independent-direct-upload-proof.json')['all546_learned_vision_norm_depthwise_patch_candidates_FP32_words_exact']
# Reconstruct immutable53 context, then invert ONLY the declared54 host delta.
seed53=load(d/'ws53-seed-source-manifest.json');ref53='aebaf00eace8fc4e512db5ca1704ae77af540123'
recon53=json.loads(subprocess.check_output(['git','-C',str(r),'show',ref53+':results/20261017/worldsense-m5-wide-cached-prefill/source-reconstruction.json']))
def bytes53(rel):
 key=('results/20261017/worldsense-m5-wide-cached-prefill/source/'+rel)if rel in recon53['overlay_files']else rel
 data=subprocess.check_output(['git','-C',str(r),'show',(ref53 if rel in recon53['overlay_files']else base)+':'+key]);assert hashlib.sha256(data).hexdigest()==seed53[rel];return data
mm=(d/'source/src/backends/metal/metal_context.mm').read_text();original_context=bytes53('src/backends/metal/metal_context.mm').decode()
def replace(a,b):
 global mm
 assert mm.count(a)==1,a;mm=mm.replace(a,b,1)
replace('#ifdef GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY\n#include "primary_isolation_shader.h"\n#endif\n','')
line='        source=[source stringByAppendingString:[NSString stringWithUTF8String:kPrimaryBlockShader]];'
replace('#ifndef GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY\n'+line+'\n#endif',line)
line='        NSString* prefillSource=[[NSString stringWithUTF8String:kPrimaryPrefillMinimalPreamble] stringByAppendingString:[NSString stringWithUTF8String:kPrimaryVectorPrefillShader]];'
replace('''#ifdef GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY
        NSString* prefillSource=[[NSString stringWithUTF8String:kPrimaryIsolatedPreamble] stringByAppendingString:[NSString stringWithUTF8String:kPrimaryBlockShader]];
        prefillSource=[prefillSource stringByAppendingString:[NSString stringWithUTF8String:kPrimaryVectorPrefillShader]];
#else
'''+line+'\n#endif',line)
replace('''#ifdef GARNET_METAL_PRIMARY_BF16_BLOCKS
        id<MTLLibrary> primaryLibrary=library;
#ifdef GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY
        primaryLibrary=prefillLibrary;
#endif
#endif
''','')
for name in ['primary_bf16_decode','primary_bf16_gemm','primary_gemm_16_64_64','primary_gemm_32_64_32','primary_expand_words','primary_expand_vector_words']:
 replace('[primaryLibrary newFunctionWithName:@"'+name+'"]','[library newFunctionWithName:@"'+name+'"]')
line='        auto function=[impl_->kernels->library newFunctionWithName:functionName constantValues:constants error:&error];'
replace('''#ifdef GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY
        auto sourceLibrary=primary?impl_->kernels->vectorPrefillLibrary:impl_->kernels->library;
        auto function=[sourceLibrary newFunctionWithName:functionName constantValues:constants error:&error];
#else
'''+line+'\n#endif',line)
assert mm==original_context,'Changes beyond source composition/lookup routing'

assert {k for k,h in closure.items()if seed53.get(k)!=h}=={'cmake/GarnetMetal.cmake','src/backends/metal/metal_context.mm','src/backends/metal/primary_isolation_shader.h','src/entry/metal_main.cpp'}
entry=(d/'source/src/entry/metal_main.cpp').read_text();marker='#ifdef GARNET_METAL_PRIMARY_FULL_ISOLATED_LIBRARY\n        info+="; primary_full_isolated_library";\n#endif\n';assert entry.count(marker)==1 and entry.replace(marker,'',1).encode()==bytes53('src/entry/metal_main.cpp')
cmake=(d/'source/cmake/GarnetMetal.cmake').read_bytes();assert cmake.startswith(bytes53('cmake/GarnetMetal.cmake'))and b'Full primary isolation requires primary weights and isolated vector prefill'in cmake
prior_program=json.loads(subprocess.check_output(['git','-C',str(r),'show',ref53+':results/20261017/worldsense-m5-wide-cached-prefill/actual-program-proof.json']))
assert all(program['programs'][mode]==prior_program['programs'][mode]for mode in ['off','primary'])
assert program['programs']['on']['main']==prior_program['programs']['off']['main']
assert len(load(d/'flag-dependency-proof.json'))==6 and all(x['expected_rejection']for x in load(d/'flag-dependency-proof.json').values())
for mode in ['on','fault']:
 assert 'Direct float upload all65536patterns/unaligned/padded/alias/source-lifetime/invalid/openbatch/Abort PASS'in(d/(mode+'-LastTest.log')).read_text()

gold=json.loads(subprocess.check_output(['git','-C',str(r),'show',base+':results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json']))
refs={(i['observation_id'],phase):(i,c)for i in gold['items']if i['policy']=='candidate'for phase,c in i['result']['model_response'].items()if isinstance(c,dict)and c.get('status')=='ok'}
query_refs={q['query']:q for q in gold['search_plans']if q['policy']=='candidate'}
storage={'201':[8401715200,6402163712,32819200]}
expected=[];parents=[];stages={};outcomes={};raw_calls={};raw_requests={}
for stage,folder in [('resource',d),('primary-speed',d/'primary-speed'),('speed',d/'speed'),('production-speed',d/'production-speed'),('confirmation',d/'confirmation')]:
    data={};calls={}
    for p in sorted(folder.glob('*.json')):
        if not p.name.startswith(('control-','candidate-')):continue
        dataset=load(p)
        if not isinstance(dataset,dict)or'items'not in dataset or'search_plans'not in dataset:continue
        label=dataset['label'];data[label]=dataset;calls[label]=[]
        assert dataset['valid']and dataset['assignments_restored']and dataset['request']==load(folder/'cases.json')
        env=load((d if stage=='confirmation'else folder)/'environment.json');worker=next(m for m in load(folder/'cohort-manifest.json')if m['label']==label)
        enabled=label.startswith('candidate');packed=enabled or stage=='primary-speed'
        assert worker['library_sha256']==env['libraries']['on'if enabled else'off']
        assert worker['serving_sha256']==(env['control_serving_sha256']if stage=='production-speed'and not enabled else closure['runtime/metal_serving.py'])
        for category,items in [('frame',dataset['items']),('query',dataset['search_plans'])]:
            for index,item in enumerate(items):
                artifact=str(p.relative_to(d));parents.append((artifact,category,str(index)));raw_requests[(artifact,category,str(index))]=item
                responses=item['result'].get('model_response',{})if category=='frame'else{'query_plan':item['result']['benchmark_model_response']}
                for phase,c in responses.items():
                    if not isinstance(c,dict)or'model_id'not in c:continue
                    assert c['status']=='ok'
                    if category=='frame':
                        ri,ref=refs[(item['observation_id'],phase)]
                        assert all(item[k]==ri[k]for k in ['image_sha256','skill_sha256'])
                        stable=lambda i:{k:v for k,v in i['result'].items()if k not in ['model_response','model_response_json']}
                        assert stable(item)==stable(ri)
                    else:
                        ri=query_refs[item['query']];ref=ri['result']['benchmark_model_response']
                        stable=lambda i:{k:v for k,v in i['result'].items()if k!='benchmark_model_response'}
                        assert stable(item)==stable(ri)
                    assert all(c[k]==ref[k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])
                    assert c['timings']['input_tokens']==ref['timings']['input_tokens']
                    tt=c['timings'];caps=['primary_bf16_blocks','primary_vector_decode','primary_kmajor_decode','primary_u32_metadata','wide_primary_storage']
                    assert all((cap in c['backend'])==packed for cap in caps)
                    assert ('primary_vector_prefill'in c['backend'])==enabled and ('lazy_resident_defaults'in c['backend'])==packed and ('cached_matrix_metadata'in c['backend'])==enabled
                    assert ('primary_prefill_isolated_library'in c['backend'])==enabled
                    assert ('primary_full_isolated_library'in c['backend'])==enabled
                    assert ('primary_duo_prefill'in c['backend'])==enabled and ('primary_minimal_prefill_library'in c['backend'])==enabled
                    assert ('direct_float_weight_upload'in c['backend'])==enabled
                    assert ('primary_vector_prefill_k32_only'in c['backend'])==enabled
                    source=env['control_source']if stage=='production-speed'and not enabled else'/Users/shawn/CantorAI/out/research/worldsense-isolated-primary/source'
                    assert tt['serving_source']==str(Path(source)/'runtime/metal_serving.py')
                    if stage=='production-speed'and not enabled:
                        assert 'dense_prefix_cache'in c['backend']and all(k not in tt for k in ['primary_storage_after','primary_encoded_delta','memory_after_decode'])
                    else:
                        assert tt['lazy_default_rehydrated_bytes']==0 and tt['memory_after_decode']['task_info_status']==0
                        owner=tt['primary_storage_after'];delta=tt['primary_encoded_delta']
                        if not packed:
                            assert tt['lazy_default_retired_bytes']==tt['lazy_default_planned_slots']==0
                            assert all(v==0 for v in owner.values())and all(v==0 for v in delta.values())
                        else:
                            count=owner['active_weights'];assert str(count)in storage
                            assert [owner['original_bytes'],owner['payload_bytes']+owner['metadata_bytes'],owner['metadata_bytes']]==storage[str(count)]
                            assert 0<owner['allocated_bytes']<owner['original_bytes']
                            assert count==201
                            assert tt['lazy_default_retired_bytes']>0 and tt['lazy_default_planned_slots']>0
                            if c['model_id']=='Qwen3.5-4B':
                                assert delta['prefill_encoded']>0 and delta['head_encoded']>0
                                assert delta['vector_decode_encoded']==delta['decode_encoded']>0 and delta['vector_head_encoded']==delta['head_encoded']
                            else:assert c['model_id']=='Qwen3-1.7B'and all(v==0 for v in delta.values())
                    identity=(artifact,category,str(index),phase,c['model_id']);calls[label].append((item,c));expected.append(identity);raw_calls[identity]=(item,c,stage)
    if not data:continue
    stages[stage]=sum(len(v)for v in calls.values())
    assert load(d/('dependencies-after-'+stage+'.json'))['all_aliases_and_file_bytes_unchanged']
    if stage=='resource':
        report=load(d/'resource-analysis.json');sampled={label:{field:max(c['timings']['memory_after_decode'][field]for _,c in values)for field in ['metal_allocated_bytes','task_physical_footprint_bytes']}for label,values in calls.items()}
        assert stages[stage]==8 and report['sampled_maximum_bytes']==sampled
        metal=1-sampled['candidate-1']['metal_allocated_bytes']/sampled['control-1']['metal_allocated_bytes'];foot=1-sampled['candidate-1']['task_physical_footprint_bytes']/sampled['control-1']['task_physical_footprint_bytes']
        assert report['passed']==(metal>=.10 and foot>=.05);outcomes[stage]=report['passed'];continue
    metrics={};walls={}
    for label,values in calls.items():
        metrics[label]={};walls[label]=statistics.median(i['native_call_seconds']for i in data[label]['items']if not i['warmup'])
        for model in ['Qwen3-1.7B','Qwen3.5-4B']:
            cc=[c for i,c in values if not i['warmup']and c['model_id']==model]
            assert len(cc)==(3 if stage=='confirmation'else 4)
            metrics[label][model]=dict(samples=len(cc),ttft=statistics.median(c['ttft_seconds']for c in cc),total=statistics.median(c['total_seconds']for c in cc),decode_tps=statistics.median((c['generated_tokens']-1)/(c['total_seconds']-c['ttft_seconds'])for c in cc))
    pairs=[('control-full-'+str(i),'candidate-full-'+str(i))for i in [1,2]]if stage=='confirmation'else[('control-1','candidate-1')]
    for control,candidate in pairs:
        report=load(folder/('analysis-'+control+'-'+candidate+'.json'if stage=='confirmation'else'analysis.json'))
        assert report['metrics']=={control:metrics[control],candidate:metrics[candidate]}
        passed=1-walls[candidate]/walls[control]>=-.02
        for model in metrics[control]:
            a,b=metrics[control][model],metrics[candidate][model]
            passed=passed and b['ttft']/a['ttft']-1<=.02 and b['total']/a['total']-1<=.02 and b['decode_tps']/a['decode_tps']-1>=-.02
        if stage=='primary-speed':
            memory={label:max(c['timings']['memory_after_decode']['metal_allocated_bytes']for _,c in calls[label])for label in [control,candidate]}
            footprint={label:max(c['timings']['memory_after_decode']['task_physical_footprint_bytes']for _,c in calls[label])for label in [control,candidate]}
            gain=1-metrics[candidate]['Qwen3.5-4B']['ttft']/metrics[control]['Qwen3.5-4B']['ttft']
            preserved=memory[candidate]/memory[control]-1<=.02 and footprint[candidate]/footprint[control]-1<=.02
            passed=passed and preserved and gain>=.05
            assert report['matched_memory_preserved']==preserved and report['observed_VLM_TTFT_gain']==gain and report['sampled_max_footprint_bytes']==footprint
        elif stage!='production-speed':
            memory={label:max(c['timings']['memory_after_decode']['metal_allocated_bytes']for _,c in calls[label])for label in [control,candidate]}
            passed=passed and 1-memory[candidate]/memory[control]>=.10
        else:assert report['sampled_max_post_request_metal_bytes'][control]is None and report['sampled_metal_reduction']is None
        assert report['passed']==passed
        outcomes[stage+(':'+control if stage=='confirmation'else'')]=passed
for name in ['calls.csv','execution-routes.csv']:
    with(d/name).open()as f:rows=list(csv.DictReader(f))
    assert sorted(tuple(row[k]for k in ['artifact','category','sample_index','phase','model_id'])for row in rows)==sorted(expected)
    assert all(row['artifact_sha256']==sha(d/row['artifact'])for row in rows)
    for row in rows:
        identity=tuple(row[k]for k in ['artifact','category','sample_index','phase','model_id']);item,c,stage=raw_calls[identity]
        assert row['warmup']==str(item['warmup'])and row['repetition']==str(item['repetition'])and row['stage']==stage
        if name=='calls.csv':
            assert all(float(row[k])==c[k]for k in ['ttft_seconds','total_seconds','generated_tokens'])
            assert float(row['derived_decode_tokens_per_second'])==(c['generated_tokens']-1)/(c['total_seconds']-c['ttft_seconds'])
            assert row['status']==c['status']and row['input_prompt_sha256']==c['input_prompt_sha256']and int(row['input_tokens'])==c['timings']['input_tokens']
            assert row['eligible_for_speed']==str(stage!='resource'and not item['warmup'])
        else:
            assert row['backend']==c['backend']and row['serving_source']==c['timings']['serving_source']
            for field in ['primary_storage_after','primary_encoded_delta','memory_after_decode']:
                assert row[field+'_available']==str(field in c['timings'])
                assert all(row[field+'_'+k]==str(v)for k,v in c['timings'].get(field,{}).items())
with(d/'skill-requests.csv').open()as f:rows=list(csv.DictReader(f))
assert sorted(tuple(row[k]for k in ['artifact','category','sample_index'])for row in rows)==sorted(parents)
for row in rows:
    item=raw_requests[tuple(row[k]for k in ['artifact','category','sample_index'])]
    assert row['artifact_sha256']==sha(d/row['artifact'])and row['warmup']==str(item['warmup'])
    assert row['native_call_seconds']==str(item.get('native_call_seconds',''))
assert load(d/'export-scope.json')['stage_calls']==stages
decision=load(d/'decision.json');assert decision['stage_outcomes']==outcomes=={'resource':True,'production-speed':False}
assert load(d/'production-speed-terminal.json')['controller_session']==20252
assert load(d/'resource-terminal.json')['controller_session']==59577
assert not any((d/stage/'analysis.json').exists()for stage in ['primary-speed','speed','confirmation'])
assert decision['prototype_confirmed']==(len(expected)==116 and all(outcomes.values())and len(outcomes)==6)
assert not decision['promote'],'Canonical production acceptance is a separate required archive'
print(json.dumps(dict(files=len(manifest),source_files=len(closure),source_overlay=10,benchmark_calls=len(expected),native_requests=len(parents),stage_outcomes=outcomes,exact_outputs=True,decision_consistent=True,scope='Independent artifact/source/call/parent-request/output/metric audit; original native tests retain their recorded numerical scope.')))
