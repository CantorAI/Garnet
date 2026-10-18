"""Independent canonical source reconstruction, native words, original goldens and decision."""
from pathlib import Path
import json,hashlib,subprocess,csv,statistics,ast,datetime
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');repo=Path('/Users/shawn/CantorAI/Garnet');d=repo/'results/20261018/worldsense-m5-sole-interleaved-mlp';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=load(d/'file-manifest.json');assert set(manifest)=={str(p.relative_to(d))for p in d.rglob('*')if p.is_file()and p.name!='file-manifest.json'}and all(sha(d/k)==v for k,v in manifest.items())
def batch(refs):
 raw=subprocess.check_output(['git','-C',str(repo),'cat-file','--batch'],input=('\n'.join(refs)+'\n').encode());offset=0;out=[]
 for _ in refs:
  end=raw.index(b'\n',offset);head=raw[offset:end].split();offset=end+1
  if head[-1]==b'missing':value=None
  else:assert head[1]==b'blob';size=int(head[2]);value=raw[offset:offset+size];offset+=size+1
  out.append(value)
 assert offset==len(raw);return out
def reconstruct(label):
 recon=load(d/(label+'source-reconstruction.json'));m=load(d/recon['source_manifest']);original=batch([recon['base_commit']+':'+rel for rel in m]);result={}
 for (rel,digest),value in zip(m.items(),original):
  if rel in recon['overlay_files']:value=(d/recon['overlay_root']/rel).read_bytes()
  assert value is not None and hashlib.sha256(value).hexdigest()==digest;result[rel]=value
 assert len(result)==recon['source_files'];return result,recon
current,recon=reconstruct('');legacy,lr=reconstruct('legacy-');assert len(current)==1720 and len(legacy)==1706
ref='59a882e9765d4c1f61cd6bc117aa620dd06abd16';prefix='results/20261017/worldsense-m5-isolated-primary/'
sm,sr=[json.loads(v)for v in batch([ref+':'+prefix+'correctness-source-manifest.json',ref+':'+prefix+'source-reconstruction.json'])];assert sm==load(d/'seed-source-manifest.json')and len(sm)==1718
seed=dict(zip(sm,batch([(ref+':'+prefix+'source/'+rel)if rel in sr['overlay_files']else sr['base_commit']+':'+rel for rel in sm])));assert all(hashlib.sha256(seed[k]).hexdigest()==v for k,v in sm.items())
delta=sorted(k for k,v in current.items()if seed.get(k)!=v);assert delta==sorted(['cmake/GarnetMetal.cmake','runtime/metal_serving.py','src/backends/metal/metal_context.h','src/backends/metal/metal_context.mm','src/backends/metal/sole_interleaved_mlp_shader.h','src/entry/metal_main.cpp','test2026/metal/sole_interleaved_test.cpp'])
for rel,ops in load(d/'source-transformations.json').items():
 text=current[rel].decode()
 for op in reversed(ops):
  if op.get('append'):assert text.endswith(op['after']);text=text[:-len(op['after'])]
  else:assert text.count(op['after'])==op['count'];text=text.replace(op['after'],op['before'])
 assert text.encode()==seed[rel],rel
assert all(current[k]==v for k,v in seed.items()if k not in load(d/'source-transformations.json'))
assert all(v==0 for v in load(d/'native-results.json').values())and len(load(d/'native-results.json'))==4
for kind,records in [('primary',186),('sole',28)]:
 words=(d/('native-off-'+kind+'_projection_words.bin')).read_bytes();assert all((d/('native-'+role+'-'+kind+'_projection_words.bin')).read_bytes()==words for role in ['primary','on','fault'])
 proof=load(d/('native-word-comparison.json'if kind=='primary'else'sole-word-comparison.json'));assert proof['records']==records
for role,programs in load(d/'actual-program-proof.json')['programs'].items():
 for kind,info in programs.items():assert sha(d/('actual-'+role+'-'+kind+'-program.metal'))==info['sha256']
assert not(d/'actual-off-sole-program.metal').read_bytes()and not(d/'actual-primary-sole-program.metal').read_bytes()
assert(d/'actual-on-sole-program.metal').read_bytes()==(d/'actual-fault-sole-program.metal').read_bytes()==current['src/backends/metal/sole_interleaved_mlp_shader.h'].decode().split('R"SOLEMLP(',1)[1].split(')SOLEMLP"',1)[0].encode()
guards=load(d/'learned-guard-adaptation-proof.json');originals=batch([ref+':'+prefix+name for name in guards])
for (name,info),original in zip(guards.items(),originals):
 assert original is not None and hashlib.sha256(original).hexdigest()==info['seed_sha256'];text=(d/name).read_text();assert hashlib.sha256(text.encode()).hexdigest()==info['current_sha256']
 text=text.replace('source1720','source1718').replace('native8tests_all4builds','native7tests_all4builds').replace('native8tests_each4builds','native7tests_each4builds')
 for op in reversed(info['operations']):assert text.count(op['after'])==op['count'];text=text.replace(op['after'],op['before'])
 assert text.replace('worldsense-sole-interleaved','worldsense-isolated-primary').encode()==original,name
corpus=load(d/'corpus-GPU-bits.json');t=corpus['totals'];assert t['packed_matrices']==201 and t['raw_small_matrices']==245 and t['HOST_raw_FP32_words_checked']==1724383232 and t['retained_primary_and_small_raw_bytes']==9850930176
assert t['GPU_original_FP32_words_checked']==t['GPU_vector_original_FP32_words_checked']==4200857600 and all(v==4200857600 for v in t['GPU_prefill_tile_FP32_words_checked'].values())
sole=[row for row in corpus['matrices']if row['GPU_sole_original_FP32_words_checked']];assert len(sole)==84 and all(row['model']=='Qwen3-1.7B'and(row['rows'],row['inner'])in[(6144,2048),(2048,6144)]and row['GPU_sole_original_FP32_words_checked']==row['rows']*row['inner']for row in sole)and sum(row['GPU_sole_original_FP32_words_checked']for row in sole)==1056964608
assert load(d/'direct-upload-corpus.json')['totals']=={'weights':546,'FP32_words_checked':334609152}
prefix_refs={row['prompt_index']:row for row in load(d/'frozen-native-guards/prefix-model-guard.json')['records']};vlm=load(d/'frozen-native-guards/state-guard.json')['records'][1]
for role in ['off','primary','on']:
 rows=load(d/('api-state-progress-'+role+'.json'))['records'];assert len(rows)==5
 for i,row in enumerate(rows):
  original=vlm if i<2 else prefix_refs[[0,1,0][i-2]];assert row['states']==original['states']and row['call']['generated_token_ids']==original['generated_token_ids']
for role in ['off','on']:assert load(d/('vision-grid-'+role+'.json'))==load(d/'frozen-native-guards/vision-grid-original.json')
assert len(load(d/'flag-dependency-proof.json'))==20 and load(d/'admission-complete.json')['twenty_expected_config_rejections']
class Normalize(ast.NodeTransformer):
 def visit_FunctionDef(self,n):return None if n.name=='native_route'else self.generic_visit(n)
for rel in ['analyze_resource.py','speed/analyze.py','primary-speed/analyze.py','production-speed/analyze.py','confirmation/analyze.py']:
 original=batch([ref+':'+prefix+rel])[0];assert original is not None
 assert ast.dump(Normalize().visit(ast.parse(original)),include_attributes=False)==ast.dump(Normalize().visit(ast.parse((d/rel).read_text().replace('worldsense-sole-interleaved','worldsense-isolated-primary'))),include_attributes=False)
gold=load(d/'oracles/worldsense-golden.json');refs={(i['observation_id'],phase):(i,c)for i in gold['items']if i['policy']=='candidate'for phase,c in i['result']['model_response'].items()if isinstance(c,dict)and c.get('status')=='ok'}
rawcalls=[];parents=[];metrics={};wall={};maxima={}
for stage,folder,labels,count in [('resource',d,['control-1','candidate-1'],2),('production-speed',d/'production-speed',['control-1','candidate-1'],5),('functional-correctness-only',d/'functional-smoke',['candidate-1'],2)]:
 path=d/'production-speed/analyze.py'if stage=='production-speed'else d/'analyze_resource.py';fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='native_route');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'archived route','exec'),ns)
 for label in labels:
  data=load(folder/(label+'.json'));assert data['valid']and data['assignments_restored']and len(data['items'])==count and not data['search_plans'];calls=[]
  for i,item in enumerate(data['items']):
   assert(item['repetition'],item['warmup'],item['policy'])==(i,i==0,'default');parents.append((stage,label,i,item))
   for phase,c in item['result']['model_response'].items():
    if not isinstance(c,dict)or c.get('status')!='ok':continue
    ri,refcall=refs[(item['observation_id'],phase)];stable=lambda x:{k:v for k,v in x.items()if k not in ['model_response','model_response_json']};assert stable(item['result'])==stable(ri['result'])
    assert all(c[k]==refcall[k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])and c['timings']['input_tokens']==refcall['timings']['input_tokens']and item['image_sha256']==ri['image_sha256']and item['skill_sha256']==ri['skill_sha256'];assert ns['native_route'](c,label=='candidate-1')
    calls.append((item,c));rawcalls.append((stage,label,i,phase,item,c))
  assert len(calls)==count*2
  if stage=='resource':maxima[label]={key:max(c['timings']['memory_after_decode'][key]for _,c in calls)for key in ['metal_allocated_bytes','task_physical_footprint_bytes']}
  if stage=='production-speed':
   metrics[label]={}
   for model in ['Qwen3-1.7B','Qwen3.5-4B']:
    cs=[c for i,c in calls if not i['warmup']and c['model_id']==model];assert len(cs)==4;metrics[label][model]=dict(samples=4,ttft=statistics.median(c['ttft_seconds']for c in cs),total=statistics.median(c['total_seconds']for c in cs),decode_tps=statistics.median((c['generated_tokens']-1)/(c['total_seconds']-c['ttft_seconds'])for c in cs))
   wall[label]=statistics.median(i['native_call_seconds']for i in data['items']if not i['warmup'])
assert len(rawcalls)==32 and len(parents)==16
resource=load(d/'resource-analysis.json');assert maxima==resource['sampled_maximum_bytes'];reductions={k:1-maxima['candidate-1'][k]/maxima['control-1'][k]for k in maxima['control-1']};assert reductions==resource['reduction']and reductions['metal_allocated_bytes']>=.10 and reductions['task_physical_footprint_bytes']>=.05 and resource['passed']
speed=load(d/'production-speed/analysis.json');assert metrics==speed['metrics']and wall==speed['native_skill_wall_p50']
comparisons=[dict(model=model,ttft_change=metrics['candidate-1'][model]['ttft']/metrics['control-1'][model]['ttft']-1,total_change=metrics['candidate-1'][model]['total']/metrics['control-1'][model]['total']-1,decode_gain=metrics['candidate-1'][model]['decode_tps']/metrics['control-1'][model]['decode_tps']-1)for model in sorted(metrics['control-1'])];assert comparisons==speed['comparisons'];gain=1-wall['candidate-1']/wall['control-1'];assert gain==speed['native_skill_wall_gain']
assert not(gain>=-.02 and all(row['ttft_change']<=.02 and row['total_change']<=.02 and row['decode_gain']>=-.02 for row in comparisons))and speed['passed']==False and speed['status']=='rejected'and speed['sampled_max_post_request_metal_bytes']['control-1']is None
calls=list(csv.DictReader((d/'calls.csv').open()));pr=list(csv.DictReader((d/'skill-requests.csv').open()));mem=list(csv.DictReader((d/'process-memory-samples.csv').open()));assert len(calls)==32 and len(pr)==16 and len(mem)==22
lookup={(row['stage'],row['worker'],int(row['repetition']),row['phase']):row for row in calls}
for stage,label,i,phase,item,c in rawcalls:
 row=lookup[(stage,label,i,phase)];assert float(row['ttft_seconds'])==c['ttft_seconds']and float(row['total_seconds'])==c['total_seconds']and int(row['generated_tokens'])==c['generated_tokens']and float(row['derived_decode_tokens_per_second'])==(c['generated_tokens']-1)/(c['total_seconds']-c['ttft_seconds'])and row['eligible_for_speed']==str(stage=='production-speed'and not item['warmup'])
inputs=load(d/'frozen-input-identities.json');assert sha(d/inputs['frame_relative'])==inputs['image_sha256']and sha(d/inputs['serialized_skill_relative'])==inputs['recorded_skill_sha256']
assert load(d/'decision.json')['decision']=='valid-reject'and not load(d/'decision.json')['promotion']
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_files=len(manifest),source_files=1720,source_overlays=len(recon['overlay_files']),legacy_files=1706,legacy_overlays=len(lr['overlay_files']),original_guard_and_all_five_metric_gate_ASTs_inverse_exact=True,original186_new28_complete_native_records_and_all_learned_words_states_tokens_vision_exact=True,all32_pipeline_goldens_native_routes_CSV_resource_speed_and_rejection_recomputed=True,calls=32,requests=16,memory_rows=22,decision='valid-reject',promotion=False)
(b/'work/worldsense-opt/ws63-archive-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
