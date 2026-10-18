"""Independent WS59 frozen-source/coverage/output/metric/archive audit, CPU only."""
from pathlib import Path
import json,hashlib,subprocess,csv,statistics,math,ast
repo=Path('/Users/shawn/CantorAI/Garnet');d=repo/'results/20261018/worldsense-m5-thread-qos';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=load(d/'file-manifest.json');assert set(manifest)=={str(p.relative_to(d))for p in d.rglob('*')if p.is_file()and p.name!='file-manifest.json'};assert all(sha(d/k)==v for k,v in manifest.items())
recon=load(d/'source-reconstruction.json');source=load(d/recon['source_manifest']);assert len(source)==1723 and len(recon['overlay_files'])==16
base=recon['base_commit']
def batch(refs):
 raw=subprocess.check_output(['git','-C',str(repo),'cat-file','--batch'],input=('\n'.join(refs)+'\n').encode());offset=0;result=[]
 for ref in refs:
  end=raw.index(b'\n',offset);header=raw[offset:end].split();offset=end+1
  if header[-1]==b'missing':value=None
  else:size=int(header[2]);assert header[1]==b'blob';value=raw[offset:offset+size];offset+=size+1
  result.append(value)
 assert offset==len(raw);return result
original=batch([base+':'+k for k in source]);current={}
for (rel,h),old in zip(source.items(),original):
 p=d/'source'/rel;value=p.read_bytes()if p.exists()else old;assert value is not None and hashlib.sha256(value).hexdigest()==h,rel;current[rel]=value
assert source==load(d/'gateway-source-manifest.json')
seedref='1a4d89b635e37c7e32a00b1673232f242ef166f1';seedpath='results/20261017/worldsense-m5-native-text-worker/'
seed_manifest=json.loads(batch([seedref+':'+seedpath+'starting-source-manifest.json'])[0]);seed_recon=json.loads(batch([seedref+':'+seedpath+'source-reconstruction.json'])[0]);assert len(seed_manifest)==1723
seed_values=batch([(seedref+':'+seedpath+'source/'+k)if k in seed_recon['overlay_files']else base+':'+k for k in seed_manifest]);seed=dict(zip(seed_manifest,seed_values));assert all(hashlib.sha256(seed[k]).hexdigest()==v for k,v in seed_manifest.items())
changed=sorted(k for k,v in current.items()if seed.get(k)!=v);assert changed==sorted(['cmake/GarnetMetal.cmake','runtime/native_text_worker.py','src/entry/metal_main.cpp','src/entry/native_text_worker.cpp','src/entry/native_worker_support.cpp','src/entry/native_worker_support.h']);assert not set(seed)-set(current)
assert current['runtime/metal_serving.py']==seed['runtime/metal_serving.py']
assert b'if(remote&&!image.IsString()&&!image.IsBin())forwardedImage=impl_->Str("")'in current['src/entry/native_text_worker.cpp']
assert load(d/'full-proof-observed-terminal.json')['session_id']==31658
assert all(load(d/'full-proof-complete.json').values())and all(load(d/'postproof-complete.json').values())
assert all(load(d/'scope-guards-result.json').values())
assert load(d/'gateway-nomodel-result.json')['scoped_QoS_closed_after_model_not_ready_error']
assert load(d/'inherited-mode-result.json')['inherited_mode_no_scopes_error_no_retry_pass']
assert all(load(d/'api-correctness-result.json').values());assert all(v==0 for v in load(d/'native-results.json').values());assert all(v['returncode']==0 for v in load(d/'resident-results.json'))
for mode in ['primary','on','fault']:assert(d/('native-'+mode+'-projection-words.bin')).read_bytes()==(d/'native-off-projection-words.bin').read_bytes()
assert load(d/'native-word-comparison.json')['records']==186
program=load(d/'actual-program-proof.json');assert program['all_four_actual_main_and_prefill_programs_byte_exact_WS54']
for mode,parts in program['programs'].items():
 for kind,info in parts.items():
  p=d/('actual-'+mode+'-'+kind+'-program.metal');assert sha(p)==info['sha256']and p.stat().st_size==info['bytes'];old=batch([seedref+':'+seedpath+p.name])[0];assert p.read_bytes()==old
corpus=load(d/'corpus-GPU-bits.json');t=corpus['totals'];assert t['packed_matrices']==201 and t['raw_small_matrices']==245 and len(corpus['matrices'])==446;assert t['GPU_original_FP32_words_checked']==t['GPU_vector_original_FP32_words_checked']==4200857600;assert t['HOST_raw_FP32_words_checked']==1724383232 and t['retained_primary_and_small_raw_bytes']==9850930176;assert set(t['GPU_prefill_tile_FP32_words_checked'])=={'32x32','64x32','64x64'}and all(v==4200857600 for v in t['GPU_prefill_tile_FP32_words_checked'].values());assert all(x['owner_accounting_restored']for x in corpus['matrices'])
assert load(d/'direct-upload-corpus.json')['totals']=={'weights':546,'FP32_words_checked':334609152}
api={mode:load(d/('api-state-progress-'+mode+'.json'))['records']for mode in ['off','primary','on']};assert all(len(x)==5 for x in api.values())
for i,a in enumerate(api['off']):
 for mode in ['primary','on']:
  b=api[mode][i];assert a['model_id']==b['model_id']and a['states']==b['states']and len(b['states'])==(32 if b['model_id']=='Qwen3.5-4B'else 28);assert all(a['call'][k]==b['call'][k]for k in ['text','generated_token_ids','input_prompt_sha256','serving_configuration'])
loaded=load(d/'invalid-image-loaded-regression.json');assert loaded['all_six_Galaxy_absent_image_kinds_exact_native_28layer_state_tokens_profile_no_retry']and len(loaded['records'])==6;ref=next(x for x in api['off']if x['model_id']=='Qwen3-1.7B')
for i,row in enumerate(loaded['records']):assert row['state']['states']==ref['states']and row['call']['generated_token_ids']==ref['call']['generated_token_ids']and row['call']['native_worker_call_index']==i+1
assert load(d/'owner-death-result.json')['all_three_parent_death_cases_pass'];assert load(d/'owner-death-active_infer/active-stack-verification.json')['actual_native_inference_stack_seen'];assert load(d/'post-owner-death-recovery.json')['exact_28layer_native_state_tokens_text_prompt_profile_after_active_owner_death_pass'];assert len(load(d/'flag-dependency-proof.json'))==10
# Reapply scoped native API guards to all retained correctness text results.
scope_tree=ast.parse((d/'qos_call_guard.py').read_text());ns={};exec(compile(scope_tree,'qos guard','exec'),ns)
for mode,rows in api.items():
 if mode=='off':continue
 for row in rows:
  if row['call']['model_id']=='Qwen3-1.7B':ns['qos_valid'](row['call'])
# Full golden, request, source, gate and counter admission independently reapplied.
gold=json.loads(batch([base+':results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json'])[0]);refs={(i['observation_id'],phase):(i,c)for i in gold['items']if i['policy']=='candidate'for phase,c in i['result']['model_response'].items()if isinstance(c,dict)and c.get('status')=='ok'}
count={};requests=0;all_calls=[]
def close(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
for stage,folder,expected_calls in [('resource',d,8),('production-speed',d/'production-speed',20)]:
 tree=ast.parse((folder/('analyze_resource.py'if stage=='resource'else'analyze.py')).read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='native_route');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'archived native admission','exec'),ns);workers={};count[stage]=0;request=load(folder/'cases.json')
 for label in ['control-1','candidate-1']:
  data=load(folder/(label+'.json'));assert data['valid']and data['assignments_restored']and data['request']==request;expected_items=2 if stage=='resource'else 5;assert len(data['items'])==expected_items and not data['search_plans'];workers[label]=[]
  for j,item in enumerate(data['items']):
   requests+=1;assert(item['repetition'],item['warmup'],item['policy'])==(j,j==0,'default')
   for phase,c in item['result']['model_response'].items():
    if not isinstance(c,dict)or c.get('status')!='ok':continue
    ri,ref=refs[(item['observation_id'],phase)];stable=lambda x:{k:v for k,v in x.items()if k not in ['model_response','model_response_json']};assert stable(item['result'])==stable(ri['result']);assert all(c[k]==ref[k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration']);assert c['timings']['input_tokens']==ref['timings']['input_tokens']and item['image_sha256']==ri['image_sha256']and item['skill_sha256']==ri['skill_sha256'];assert ns['native_route'](c,label=='candidate-1');count[stage]+=1;workers[label].append((item,c));all_calls.append((stage,label,item,c))
  assert len(workers[label])==expected_items*2
 assert count[stage]==expected_calls
 if stage=='resource':
  analysis=load(d/'resource-analysis.json');maxima={label:{key:max(c['timings']['memory_after_decode'][key]for _,c in rows)for key in ['metal_allocated_bytes','task_physical_footprint_bytes']}for label,rows in workers.items()};assert maxima==analysis['sampled_maximum_bytes'];reduction={k:1-maxima['candidate-1'][k]/maxima['control-1'][k]for k in maxima['control-1']};assert analysis['passed']and reduction['metal_allocated_bytes']>=.1 and reduction['task_physical_footprint_bytes']>=.05
  for k,v in reduction.items():close(v,analysis['reduction'][k])
 else:
  a=load(folder/'analysis.json');metrics={}
  for label,rows in workers.items():
   metrics[label]={}
   for model in ['Qwen3.5-4B','Qwen3-1.7B']:
    cs=[c for item,c in rows if not item['warmup']and c['model_id']==model];assert len(cs)==4;v=dict(samples=4,ttft=statistics.median(c['ttft_seconds']for c in cs),total=statistics.median(c['total_seconds']for c in cs),decode_tps=statistics.median((c['generated_tokens']-1)/(c['total_seconds']-c['ttft_seconds'])for c in cs));metrics[label][model]=v
    for k,x in v.items():close(x,a['metrics'][label][model][k])
  walls={label:statistics.median(item['native_call_seconds']for item in load(folder/(label+'.json'))['items']if not item['warmup'])for label in workers};wallgain=1-walls['candidate-1']/walls['control-1'];close(wallgain,a['native_skill_wall_gain']);passed=wallgain>=-.02
  for comparison in a['comparisons']:
   model=comparison['model'];u,v=metrics['control-1'][model],metrics['candidate-1'][model];delta=dict(ttft_change=v['ttft']/u['ttft']-1,total_change=v['total']/u['total']-1,decode_gain=v['decode_tps']/u['decode_tps']-1)
   for k,x in delta.items():close(x,comparison[k])
   passed=passed and delta['ttft_change']<=.02 and delta['total_change']<=.02 and delta['decode_gain']>=-.02
  assert not passed and not a['passed'];assert metrics['candidate-1']['Qwen3-1.7B']['decode_tps']/metrics['control-1']['Qwen3-1.7B']['decode_tps']-1<-.02
  assert a['sampled_max_post_request_metal_bytes']['control-1']is None
# Candidate-only functional smoke is correctness evidence, never speed samples.
folder=d/'functional-smoke';functional=load(folder/'candidate-1.json');assert functional['valid']and functional['assignments_restored']and len(functional['items'])==2
assert load(folder/'functional-verification.json')['speed_eligibility']is False
native_tree=ast.parse((d/'analyze_resource.py').read_text());fn=next(x for x in native_tree.body if isinstance(x,ast.FunctionDef)and x.name=='native_route');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'scope admission','exec'),ns)
functional_calls=0
for item in functional['items']:
 for phase,c in item['result']['model_response'].items():
  if not isinstance(c,dict)or c.get('status')!='ok':continue
  ri,ref=refs[(item['observation_id'],phase)];assert all(c[k]==ref[k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration']);assert ns['native_route'](c,True);functional_calls+=1
assert functional_calls==4
csvcalls=list(csv.DictReader((d/'calls.csv').open()));csvparents=list(csv.DictReader((d/'skill-requests.csv').open()));assert len(csvcalls)==32 and len(csvparents)==16 and requests==14
assert {stage:sum(x['stage']==stage for x in csvcalls)for stage in ['resource','production-speed','functional-correctness-only']}=={'resource':8,'production-speed':20,'functional-correctness-only':4};assert all(x['eligible_for_speed']=='False'for x in csvcalls if x['stage']in ['resource','functional-correctness-only']);assert len([x for x in csvcalls if x['eligible_for_speed']=='True'])==16
for stage,label,item,c in all_calls:
 row=next(x for x in csvcalls if x['stage']==stage and x['worker']==label and int(x['repetition'])==item['repetition']and x['model_id']==c['model_id']);close(float(row['ttft_seconds']),c['ttft_seconds']);close(float(row['total_seconds']),c['total_seconds']);close(float(row['derived_decode_tokens_per_second']),(c['generated_tokens']-1)/(c['total_seconds']-c['ttft_seconds']))
decision=load(d/'decision.json');assert decision['decision']=='reject'and not decision['promote']and not decision['later_stages_dispatched']and decision['no_unchanged_retry'];assert not any((d/f/'analysis.json').exists()for f in ['primary-speed','speed','confirmation']);assert load(d/'restoration-verification.json')['benchmark_valid_request_http404'];assert load(d/'dependencies-after-resource.json')['all_aliases_and_file_bytes_unchanged']and load(d/'dependencies-after-production-speed.json')['all_aliases_and_file_bytes_unchanged'];assert load(d/'cohort-native-identity-audit.json')['exact_two_child_calls_no_hidden_retry']and load(d/'production-speed/cohort-native-identity-audit.json')['exact_five_child_calls_no_hidden_retry']
frozen=load(d/'frozen-input-identities.json');assert sha(d/frozen['frame_relative'])==frozen['image_sha256'];assert frozen['exact_serialized_skill_reconstructed'];assert sha(d/frozen['serialized_skill_relative'])==frozen['recorded_skill_sha256'];assert load(d/frozen['serialized_skill_relative'])==load(d/'control-1.json')['items'][0]['skill'];assert sha(d/frozen['source_yaml_relative'])==frozen['source_yaml_sha256']
# The archive must never contain the private auth helper or copies of it.
assert not any(p.name=='worldsense-http.py'for p in d.rglob('*'))
print(json.dumps(dict(archive=str(d),verified_SHA_files=len(manifest),source_files=len(source),source_overlay=len(recon['overlay_files']),seed_changes=len(changed),valid_benchmark_model_calls=28,functional_correctness_only_calls=4,total_export_calls=32,benchmark_parent_requests=14,total_export_parent_requests=16,all_numerical_output_metric_CSV_decision_checks_pass=True),indent=2))
