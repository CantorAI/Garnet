"""Independent WS60 frozen-source/coverage/output/metric/archive audit, CPU only."""
from pathlib import Path
import json,hashlib,subprocess,csv,statistics,math,ast
repo=Path('/Users/shawn/CantorAI/Garnet');d=repo/'results/20261018/worldsense-m5-dense-prefetch';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=load(d/'file-manifest.json');assert set(manifest)=={str(p.relative_to(d))for p in d.rglob('*')if p.is_file()and p.name!='file-manifest.json'};assert all(sha(d/k)==v for k,v in manifest.items())
recon=load(d/'source-reconstruction.json');source=load(d/recon['source_manifest']);assert len(source)==1720 and len(recon['overlay_files'])==13
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
assert source==load(d/'correctness-source-manifest.json')
seedref='59a882e9765d4c1f61cd6bc117aa620dd06abd16';seedpath='results/20261017/worldsense-m5-isolated-primary/'
seed_manifest=json.loads(batch([seedref+':'+seedpath+'correctness-source-manifest.json'])[0]);seed_recon=json.loads(batch([seedref+':'+seedpath+'source-reconstruction.json'])[0]);assert len(seed_manifest)==1718
seed_values=batch([(seedref+':'+seedpath+'source/'+k)if k in seed_recon['overlay_files']else base+':'+k for k in seed_manifest]);seed=dict(zip(seed_manifest,seed_values));assert all(hashlib.sha256(seed[k]).hexdigest()==v for k,v in seed_manifest.items())
changed=sorted(k for k,v in current.items()if seed.get(k)!=v)
assert changed==sorted(['cmake/GarnetMetal.cmake','runtime/metal_serving.py','src/backends/metal/dense_prefetch_shader.h','src/backends/metal/metal_context.h','src/backends/metal/metal_context.mm','src/entry/metal_main.cpp','test2026/metal/dense_prefetch_test.cpp']);assert not set(seed)-set(current)
assert all(load(d/'full-proof-complete.json').values())
assert load(d/'postproof-source-admission.json')['observed_controller_session']==68305
assert load(d/'postproof-source-admission.json')['original_Weight_loader_and_wide_scope_source_exact54']
assert load(d/'model-assets-after-screen.json')['all_recorded_model_assets_rehashed_exact']
assert len(load(d/'flag-dependency-proof.json'))==10
assert all(x['expected_rejection']for x in load(d/'flag-dependency-proof.json').values())
assert load(d/'admission-complete.json')['current_forced_raw_prefetch_bounds_fallback_bits_abort_guards_pass']
for rel in ['src/backends/metal/body_shape_shader.h','test2026/metal/primary_projection_test.cpp']:
 assert current[rel]==seed[rel]
import re,struct
literal=lambda x:re.findall(rb'R"([A-Za-z0-9_]*)\((.*?)\)\1"',x,re.S)
assert literal(current['src/backends/metal/metal_context.mm'])==literal(seed['src/backends/metal/metal_context.mm'])
region=lambda x:x[x.index(b'MetalBF16Weight MetalContext::UploadBF16('):x.index(b'void MetalContext::EncodeArgmax(')]
assert region(current['src/backends/metal/metal_context.mm'])==region(seed['src/backends/metal/metal_context.mm'])
assert all(load(d/'api-correctness-result.json').values());assert all(v==0 for v in load(d/'native-results.json').values());assert all(v['returncode']==0 for v in load(d/'resident-results.json'))
for mode in ['primary','on','fault']:assert(d/('native-'+mode+'-projection-words.bin')).read_bytes()==(d/'native-off-projection-words.bin').read_bytes()
assert load(d/'native-word-comparison.json')['records']==186
for mode in ['primary','on','fault']:assert(d/('native-'+mode+'-dense-prefetch-words.bin')).read_bytes()==(d/'native-off-dense-prefetch-words.bin').read_bytes()
words=(d/'native-off-dense-prefetch-words.bin').read_bytes();assert words==(d/'native-off-projection-words.bin').read_bytes()
assert hashlib.sha256(words).hexdigest()==load(d/'dense-prefetch-word-comparison.json')['sha256']
offset=0;records=0
while offset<len(words):
 rows,inner,columns,pass_index=struct.unpack_from('<QQQQ',words,offset);offset+=32
 assert rows in [1,2,3,16,32,64]and pass_index in [0,1]and inner>0 and columns>0
 offset+=rows*columns*4;records+=1
assert offset==len(words)and records==186

program=load(d/'actual-program-proof.json');assert program['OFF_main_exact54']and program['other_main_exact54_after_removing_one_exact_new_fragment']and program['all_four_prefill_programs_exact54']
for mode,parts in program['programs'].items():
 for kind,info in parts.items():
  p=d/('actual-'+mode+'-'+kind+'-program.metal');assert sha(p)==info['sha256']and p.stat().st_size==info['bytes'];old=batch([seedref+':'+seedpath+p.name])[0]
  value=p.read_bytes();fragment=current['src/backends/metal/dense_prefetch_shader.h'].split(b'R\"DENSELOOK(',1)[1].split(b')DENSELOOK\"',1)[0]
  if kind=='main'and mode!='off':assert value.count(fragment)==1 and value.replace(fragment,b'',1)==old
  else:assert value==old
corpus=load(d/'corpus-GPU-bits.json');t=corpus['totals'];assert t['packed_matrices']==201 and t['raw_small_matrices']==245 and len(corpus['matrices'])==446;assert t['GPU_original_FP32_words_checked']==t['GPU_vector_original_FP32_words_checked']==4200857600;assert t['HOST_raw_FP32_words_checked']==1724383232 and t['retained_primary_and_small_raw_bytes']==9850930176;assert set(t['GPU_prefill_tile_FP32_words_checked'])=={'32x32','64x32','64x64'}and all(v==4200857600 for v in t['GPU_prefill_tile_FP32_words_checked'].values());assert all(x['owner_accounting_restored']for x in corpus['matrices'])
assert load(d/'direct-upload-corpus.json')['totals']=={'weights':546,'FP32_words_checked':334609152}
api={mode:load(d/('api-state-progress-'+mode+'.json'))['records']for mode in ['off','primary','on']};assert all(len(x)==5 for x in api.values())
for i,a in enumerate(api['off']):
 for mode in ['primary','on']:
  b=api[mode][i];assert a['model_id']==b['model_id']and a['states']==b['states']and len(b['states'])==(32 if b['model_id']=='Qwen3.5-4B'else 28);assert all(a['call'][k]==b['call'][k]for k in ['text','generated_token_ids','input_prompt_sha256','serving_configuration'])
for mode,rows in api.items():
 for row in rows:
  call=row['call'];value=call['timings']['dense_prefetch_encoded_delta']
  assert isinstance(value,int)and not isinstance(value,bool)and value>=0
  assert ('dense_raw_decode_prefetch'in call['backend'])==(mode!='off')
  assert (value>0)==(mode!='off'and call['model_id']=='Qwen3-1.7B')
# The bundled original native oracles match frozen external inputs and all states.
inputs=load(d/'frozen-native-guard-identities.json')['files'];external=load(d/'full-proof-identities-before.json')['external_inputs']
for rel,info in inputs.items():assert sha(d/rel)==info['sha256']==external[info['original_path']]
vlref=load(d/'frozen-native-guards/state-guard.json')['records'][1]
textrefs={x['prompt_index']:x for x in load(d/'frozen-native-guards/prefix-model-guard.json')['records']}
for rows in api.values():
 for i,row in enumerate(rows):
  ref=vlref if i<2 else textrefs[[0,1,0][i-2]]
  assert row['states']==ref['states']and row['call']['generated_token_ids']==ref['call']['generated_token_ids']
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
decision=load(d/'decision.json');assert decision['decision']=='reject'and not decision['promote']and not decision['later_stages_dispatched']and decision['no_unchanged_retry'];assert not any((d/f/'analysis.json').exists()for f in ['primary-speed','speed','confirmation']);assert load(d/'restoration-verification.json')['benchmark_valid_request_http404'];assert load(d/'dependencies-after-resource.json')['all_aliases_and_file_bytes_unchanged']and load(d/'dependencies-after-production-speed.json')['all_aliases_and_file_bytes_unchanged']
frozen=load(d/'frozen-input-identities.json');assert sha(d/frozen['frame_relative'])==frozen['image_sha256'];assert frozen['exact_serialized_skill_reconstructed'];assert sha(d/frozen['serialized_skill_relative'])==frozen['recorded_skill_sha256'];assert load(d/frozen['serialized_skill_relative'])==load(d/'control-1.json')['items'][0]['skill'];assert sha(d/frozen['source_yaml_relative'])==frozen['source_yaml_sha256']
snapshots=list(csv.DictReader((d/'process-memory-snapshots.csv').open()));assert len(snapshots)==22
for stage,label,item,c in all_calls:
 if 'memory_after_decode'not in c['timings']:continue
 row=next(x for x in snapshots if x['stage']==stage and x['worker']==label and int(x['repetition'])==item['repetition']and x['model_id']==c['model_id'])
 for key in ['metal_allocated_bytes','task_physical_footprint_bytes','task_resident_bytes','task_compressed_bytes','task_internal_bytes','task_virtual_bytes']:assert int(row[key])==c['timings']['memory_after_decode'][key]
# The archive must never contain the private auth helper or copies of it.
assert not any(p.name=='worldsense-http.py'for p in d.rglob('*'))
print(json.dumps(dict(archive=str(d),verified_SHA_files=len(manifest),source_files=len(source),source_overlay=len(recon['overlay_files']),seed_changes=len(changed),valid_benchmark_model_calls=28,functional_correctness_only_calls=4,total_export_calls=32,benchmark_parent_requests=14,total_export_parent_requests=16,all_numerical_output_metric_CSV_decision_checks_pass=True),indent=2))
