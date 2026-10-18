"""Independent archive reconstruction, original outputs and gate recomputation."""
from pathlib import Path
import json, hashlib, subprocess, ast, csv, statistics, math, datetime
repo=Path('/Users/shawn/CantorAI/Garnet');d=repo/'results/20261018/worldsense-m5-validated-text-worker'
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=load(d/'file-manifest.json');assert set(manifest)=={str(p.relative_to(d))for p in d.rglob('*')if p.is_file()and p.name!='file-manifest.json'}
assert all(sha(d/k)==v for k,v in manifest.items())
def batch(refs):
 raw=subprocess.check_output(['git','-C',str(repo),'cat-file','--batch'],input=('\n'.join(refs)+'\n').encode());offset=0;out=[]
 for _ in refs:
  end=raw.index(b'\n',offset);head=raw[offset:end].split();offset=end+1
  if head[-1]==b'missing':v=None
  else:
   assert head[1]==b'blob';size=int(head[2]);v=raw[offset:offset+size];offset+=size+1
  out.append(v)
 assert offset==len(raw);return out
def reconstruct(manifest,recon,folder):
 old=batch([recon['base_commit']+':'+key for key in manifest]);result={}
 for (key,value),original in zip(manifest.items(),old):
  p=folder/key;current=p.read_bytes()if key in recon['overlay_files']else original
  assert current is not None and hashlib.sha256(current).hexdigest()==value,key
  result[key]=current
 assert {str(p.relative_to(folder))for p in folder.rglob('*')if p.is_file()}==set(recon['overlay_files'])
 return result
current=reconstruct(load(d/'gateway-source-manifest.json'),load(d/'source-reconstruction.json'),d/'source')
legacy=reconstruct(load(d/'validated-source-manifest.json'),load(d/'legacy-source-reconstruction.json'),d/'legacy-source')
assert len(current)==1726 and len(legacy)==1706
seedref='1a4d89b635e37c7e32a00b1673232f242ef166f1';seedpath='results/20261017/worldsense-m5-native-text-worker/'
sm,sr=[json.loads(x)for x in batch([seedref+':'+seedpath+'starting-source-manifest.json',seedref+':'+seedpath+'source-reconstruction.json'])]
assert sm==load(d/'parent-seed-source-manifest.json')and len(sm)==1723
seed_values=batch([(seedref+':'+seedpath+'source/'+key)if key in sr['overlay_files']else sr['base_commit']+':'+key for key in sm]);seed=dict(zip(sm,seed_values))
assert all(hashlib.sha256(seed[key]).hexdigest()==value for key,value in sm.items())
changed=sorted(key for key,value in current.items()if seed.get(key)!=value)
assert changed==sorted(['cmake/GarnetMetal.cmake','src/entry/metal_main.cpp','src/entry/native_text_worker.cpp','src/entry/validated_worker_host.mm','runtime/validated_text_worker.py','runtime/validated_ws10/metal_serving.py'])
assert not set(seed)-set(current)
overlay=current['runtime/validated_ws10/metal_serving.py'];extra=b'native_model_start_monotonic=start, native_first_token_monotonic=first, '
assert overlay.count(extra)==1 and overlay.replace(extra,b'')==legacy['runtime/metal_serving.py']
marker=b'#ifdef GARNET_METAL_NATIVE_TEXT_WORKER_VALIDATED_RAW\n        info+="; native_text_worker_validated_ws10";\n#endif\n'
assert current['src/entry/metal_main.cpp'].replace(marker,b'')==seed['src/entry/metal_main.cpp']
a=seed['src/entry/native_text_worker.cpp'];b=current['src/entry/native_text_worker.cpp'];suffix=b'std::string NativeTextWorker::Infer('
assert a[a.index(suffix):]==b[b.index(suffix):]
for start,end in [(b' X::Value Call(',b' void Stop('),(b' void Stop(',b' J Memory('),(b' J Memory(){',b'};\nstd::recursive_mutex&'),(b'std::string NativeTextWorker::Serve(',b'std::string NativeTextWorker::Infer(')]:
 assert a[a.index(start):a.index(end,a.index(start))]==b[b.index(start):b.index(end,b.index(start))]
assert current['src/entry/validated_worker_host.mm']==(d/'native-host/worker_host.mm').read_bytes()
assert all(current[key]==value for key,value in seed.items()if key.startswith(('src/backends/','xModel/','src/image/','test2026/')))
assert all(v==0 for v in load(d/'native-results.json').values())
words=(d/'native-off-projection-words.bin').read_bytes()
assert all((d/('native-'+mode+'-projection-words.bin')).read_bytes()==words for mode in ['primary','on','fault'])
assert(d/'legacy-projection-words.bin').read_bytes()==words and load(d/'native-word-comparison.json')['records']==186
t=load(d/'corpus-GPU-bits.json')['totals'];assert t['packed_matrices']==201 and t['raw_small_matrices']==245 and t['HOST_raw_FP32_words_checked']==1724383232 and t['retained_primary_and_small_raw_bytes']==9850930176
assert t['GPU_original_FP32_words_checked']==t['GPU_vector_original_FP32_words_checked']==4200857600
assert set(t['GPU_prefill_tile_FP32_words_checked'])=={'32x32','64x32','64x64'}and all(v==4200857600 for v in t['GPU_prefill_tile_FP32_words_checked'].values())
assert load(d/'direct-upload-corpus.json')['totals']=={'weights':546,'FP32_words_checked':334609152}
assert all(x['returncode']==0 for x in load(d/'resident-results.json'))
api={mode:load(d/('api-state-progress-'+mode+'.json'))['records']for mode in ['off','primary','on']};assert all(len(x)==5 for x in api.values())
prefix={x['prompt_index']:x for x in load(d/'frozen-native-guards/prefix-model-guard.json')['records']}
refvlm=load(d/'frozen-native-guards/state-guard.json')['records'][1]
for mode,rows in api.items():
 for i,row in enumerate(rows):
  assert row['states']==api['off'][i]['states']and all(row['call'][key]==api['off'][i]['call'][key]for key in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])
  ref=refvlm if i<2 else prefix[[0,1,0][i-2]]
  assert row['states']==ref['states']and row['call']['generated_token_ids']==ref['generated_token_ids']
for mode in ['off','on']:assert load(d/('vision-grid-'+mode+'.json'))==load(d/'frozen-native-guards/vision-grid-original.json')
loaded=load(d/'invalid-image-loaded-regression.json');assert len(loaded['records'])==6
textref=next(x for x in load(d/'oracles/api-state-guard-off.json')['records']if x['model_id']=='Qwen3-1.7B')
for i,row in enumerate(loaded['records']):
 assert row['state']['states']==textref['states']and row['call']['native_worker_call_index']==i+1
 assert all(row['call'][key]==textref['call'][key]for key in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])
assert load(d/'owner-death-result.json')['all_three_parent_death_cases_pass']and load(d/'owner-death-active_infer/active-stack-verification.json')['actual_native_inference_stack_seen']
recovery=load(d/'post-owner-death-recovery.json');assert recovery['state']['states']==textref['states']and recovery['call']['generated_token_ids']==textref['call']['generated_token_ids']
assert len(load(d/'flag-dependency-proof.json'))==13
gold=load(d/'oracles/worldsense-golden.json');refs={(i['observation_id'],phase):(i,c)for i in gold['items']if i['policy']=='candidate'for phase,c in i['result']['model_response'].items()if isinstance(c,dict)and c.get('status')=='ok'}
stage_calls={};nrequests=0
def close(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
for stage,folder,total in [('resource',d,8),('production-speed',d/'production-speed',20),('functional-correctness-only',d/'functional-smoke',4)]:
 tree=ast.parse((d/'analyze_resource.py'if stage!='production-speed'else folder/'analyze.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='native_route');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'archived admission','exec'),ns)
 workers={};stage_calls[stage]=0
 labels=['candidate-1']if stage=='functional-correctness-only'else['control-1','candidate-1']
 for label in labels:
  data=load(folder/(label+'.json'));assert data['valid']and data['assignments_restored']and not data['search_plans']and data['request']==load(folder/'cases.json')
  assert len(data['items'])==(5 if stage=='production-speed'else 2);workers[label]=[]
  for j,item in enumerate(data['items']):
   nrequests+=1;assert(item['repetition'],item['warmup'],item['policy'])==(j,j==0,'default')
   for phase,c in item['result']['model_response'].items():
    if not isinstance(c,dict)or c.get('status')!='ok':continue
    ri,ref=refs[(item['observation_id'],phase)];stable=lambda x:{k:v for k,v in x.items()if k not in ['model_response','model_response_json']}
    assert stable(item['result'])==stable(ri['result'])and all(c[key]==ref[key]for key in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])
    assert c['timings']['input_tokens']==ref['timings']['input_tokens']and item['image_sha256']==ri['image_sha256']and item['skill_sha256']==ri['skill_sha256']
    assert ns['native_route'](c,label=='candidate-1')
    if label=='candidate-1'and c['model_id']=='Qwen3-1.7B':
     child=c['timings']['native_gateway']['child'];assert child['model_backend']['library_sha256']=='dc3d135d664cde077a4333bfeea47252ff83ac64061c3fea562fdac6ccfdac72'
    workers[label].append((item,c));stage_calls[stage]+=1
 assert stage_calls[stage]==total
 if stage=='resource':
  analysis=load(d/'resource-analysis.json');maxima={label:{key:max(c['timings']['memory_after_decode'][key]for _,c in rows)for key in ['metal_allocated_bytes','task_physical_footprint_bytes']}for label,rows in workers.items()}
  assert maxima==analysis['sampled_maximum_bytes']and analysis['passed']
  for key in maxima['control-1']:close(1-maxima['candidate-1'][key]/maxima['control-1'][key],analysis['reduction'][key])
  assert analysis['reduction']['metal_allocated_bytes']>=.1 and analysis['reduction']['task_physical_footprint_bytes']>=.05
 if stage=='production-speed':
  analysis=load(folder/'analysis.json');metrics={}
  for label,rows in workers.items():
   metrics[label]={}
   for model in ['Qwen3.5-4B','Qwen3-1.7B']:
    cs=[c for item,c in rows if not item['warmup']and c['model_id']==model];assert len(cs)==4
    m=dict(samples=4,ttft=statistics.median(c['ttft_seconds']for c in cs),total=statistics.median(c['total_seconds']for c in cs),decode_tps=statistics.median((c['generated_tokens']-1)/(c['total_seconds']-c['ttft_seconds'])for c in cs));metrics[label][model]=m
    for key,value in m.items():close(value,analysis['metrics'][label][model][key])
  walls={label:statistics.median(i['native_call_seconds']for i in load(folder/(label+'.json'))['items']if not i['warmup'])for label in workers}
  gain=1-walls['candidate-1']/walls['control-1'];close(gain,analysis['native_skill_wall_gain']);passed=gain>=-.02
  for row in analysis['comparisons']:
   a=metrics['control-1'][row['model']];b=metrics['candidate-1'][row['model']];diff=dict(ttft_change=b['ttft']/a['ttft']-1,total_change=b['total']/a['total']-1,decode_gain=b['decode_tps']/a['decode_tps']-1)
   for key,value in diff.items():close(value,row[key])
   passed=passed and diff['ttft_change']<=.02 and diff['total_change']<=.02 and diff['decode_gain']>=-.02
  assert not passed and not analysis['passed']and analysis['sampled_max_post_request_metal_bytes']['control-1']is None
assert nrequests==16
calls=list(csv.DictReader((d/'calls.csv').open()));parents=list(csv.DictReader((d/'skill-requests.csv').open()));samples=list(csv.DictReader((d/'process-memory-samples.csv').open()))
assert len(calls)==32 and len(parents)==16 and len(samples)==40
assert {stage:sum(c['stage']==stage for c in calls)for stage in stage_calls}==stage_calls
assert all(c['eligible_for_speed']==str(c['stage']=='production-speed'and c['warmup']=='False')for c in calls)
inputs=load(d/'frozen-input-identities.json');assert sha(d/inputs['frame_relative'])==inputs['image_sha256']and sha(d/inputs['serialized_skill_relative'])==inputs['recorded_skill_sha256']
assert load(d/inputs['serialized_skill_relative'])==load(d/'control-1.json')['items'][0]['skill']
for rel,info in load(d/'frozen-native-guard-identities.json')['files'].items():
 assert sha(d/rel)==info['sha256']==load(d/'learned-input-identities-before.json')['external'][info['original_path']]
assert load(d/'decision.json')['decision']=='reject'and not load(d/'decision.json')['promote']
assert load(d/'resource-independent-audit.json')['no_new_inference_or_timing_retry']
assert load(d/'acceptance-gate-audit.json')['exact_model_role_serving_path_assertions_inverse_normalized']
assert load(d/'model-assets-after-screen.json')['files']==19
assert not any((d/stage/'analysis.json').exists()for stage in ['primary-speed','speed','confirmation'])
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_files=len(manifest),parent_files=1726,legacy_files=1706,parent_seed_changed=changed,all_sources_goldens_numerics_states_tokens_native_routes_original_resource_and_speed_metrics_recomputed=True,stage_calls=stage_calls,requests=nrequests,CSV_calls=32,CSV_memory_process_rows=40,decision='valid_reject',scope='Archive/source/correctness/metric consistency, not fresh inference, independent timing or full goal completion.')
(Path(__file__).parent/'ws61-archive-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
