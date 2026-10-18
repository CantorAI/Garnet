"""Independent source reconstruction, full output/word and raw-counter recomputation."""
from pathlib import Path
import json,hashlib,subprocess,csv,collections,math,datetime
repo=Path('/Users/shawn/CantorAI/Garnet');d=repo/'results/20261018/worldsense-m5-gpu-pass-profile';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=load(d/'file-manifest.json');assert set(manifest)=={str(p.relative_to(d))for p in d.rglob('*')if p.is_file()and p.name!='file-manifest.json'}and all(sha(d/k)==v for k,v in manifest.items())
def batch(refs):
 raw=subprocess.check_output(['git','-C',str(repo),'cat-file','--batch'],input=('\n'.join(refs)+'\n').encode());offset=0;result=[]
 for _ in refs:
  end=raw.index(b'\n',offset);header=raw[offset:end].split();offset=end+1
  if header[-1]==b'missing':value=None
  else:
   assert header[1]==b'blob';size=int(header[2]);value=raw[offset:offset+size];offset+=size+1
  result.append(value)
 assert offset==len(raw);return result
recon=load(d/'source-reconstruction.json');source_manifest=load(d/recon['source_manifest']);assert len(source_manifest)==1720
old=batch([recon['base_commit']+':'+rel for rel in source_manifest]);current={}
for (rel,digest),original in zip(source_manifest.items(),old):
 value=(d/'source'/rel).read_bytes()if rel in recon['overlay_files']else original;assert value is not None and hashlib.sha256(value).hexdigest()==digest;current[rel]=value
seedref='59a882e9765d4c1f61cd6bc117aa620dd06abd16';seedpath='results/20261017/worldsense-m5-isolated-primary/'
sm,sr=[json.loads(value)for value in batch([seedref+':'+seedpath+'correctness-source-manifest.json',seedref+':'+seedpath+'source-reconstruction.json'])]
assert len(sm)==1718 and sm==load(d/'seed-source-manifest.json')
seed_values=batch([(seedref+':'+seedpath+'source/'+rel)if rel in sr['overlay_files']else sr['base_commit']+':'+rel for rel in sm]);seed=dict(zip(sm,seed_values));assert all(hashlib.sha256(seed[k]).hexdigest()==v for k,v in sm.items())
changed=sorted(k for k,v in current.items()if seed.get(k)!=v)
assert changed==sorted(['cmake/GarnetMetal.cmake','src/backends/metal/metal_context.h','src/backends/metal/metal_context.mm','src/backends/metal/gpu_pass_counter_trace.inc','src/entry/metal_main.cpp','runtime/metal_serving.py','test2026/metal/gpu_pass_counter_test.cpp'])
for rel,operations in load(d/'source-transformations.json').items():
 text=current[rel].decode()
 for item in reversed(operations):
  if item.get('append'):assert text.endswith(item['after']);text=text[:-len(item['after'])]
  else:assert text.count(item['after'])==item['count'];text=text.replace(item['after'],item['before'])
 assert text.encode()==seed[rel],rel
assert all(current[rel]==value for rel,value in seed.items()if rel not in load(d/'source-transformations.json'))
assert b'#define PassTraceEncoder(state,name) [(state).command computeCommandEncoder]'in current['src/backends/metal/gpu_pass_counter_trace.inc']
assert load(d/'defaultOFF-preprocessor-proof.json')['diagnostic_label_arguments_and_wrapper_absent']
assert len(load(d/'native-results.json'))==5 and all(v==0 for v in load(d/'native-results.json').values())
words=(d/'native-off-projection-words.bin').read_bytes();assert all((d/('native-'+mode+'-projection-words.bin')).read_bytes()==words for mode in ['primary','on','quiet','fault'])
assert load(d/'native-word-comparison.json')['records']==186
for mode,parts in load(d/'actual-program-proof.json')['programs'].items():
 for kind,info in parts.items():assert sha(d/('actual-'+mode+'-'+kind+'-program.metal'))==info['sha256']
corpus=load(d/'corpus-GPU-bits.json');t=corpus['totals'];assert len(corpus['matrices'])==446 and t['packed_matrices']==201 and t['raw_small_matrices']==245 and t['HOST_raw_FP32_words_checked']==1724383232 and t['retained_primary_and_small_raw_bytes']==9850930176
assert t['GPU_original_FP32_words_checked']==t['GPU_vector_original_FP32_words_checked']==4200857600 and all(v==4200857600 for v in t['GPU_prefill_tile_FP32_words_checked'].values())
assert load(d/'direct-upload-corpus.json')['totals']=={'weights':546,'FP32_words_checked':334609152}
api={mode:load(d/('api-state-progress-'+mode+'.json'))['records']for mode in ['off','primary','on','quiet']};assert all(len(v)==5 for v in api.values())
prefix={v['prompt_index']:v for v in load(d/'frozen-native-guards/prefix-model-guard.json')['records']};vlm=load(d/'frozen-native-guards/state-guard.json')['records'][1]
for mode,rows in api.items():
 for index,row in enumerate(rows):
  ref=vlm if index<2 else prefix[[0,1,0][index-2]];assert row['states']==ref['states']and row['call']['generated_token_ids']==ref['generated_token_ids']
  assert all(row['call'][key]==api['off'][index]['call'][key]for key in ['text','generated_token_ids','input_prompt_sha256','serving_configuration'])
  assert ('counter_trace_request_id'in row['call']['timings'])==(mode!='quiet')
for mode in ['off','on']:assert load(d/('vision-grid-'+mode+'.json'))==load(d/'frozen-native-guards/vision-grid-original.json')
fault=load(d/'counter-fault-model-recovery.json');textref=prefix[0];assert fault['injected_call']['status']=='error'and'Injected GPU trace resolution failure'in fault['injected_call']['error_message'];assert fault['states']==textref['states']and fault['recovery_call']['generated_token_ids']==textref['generated_token_ids']
gold=load(d/'oracles/worldsense-golden.json');refs={(item['observation_id'],phase):(item,c)for item in gold['items']if item['policy']=='candidate'for phase,c in item['result']['model_response'].items()if isinstance(c,dict)and c.get('status')=='ok'}
data=load(d/'capture/candidate-1.json');assert data['valid']and data['assignments_restored']and len(data['items'])==2 and not data['search_plans']and data['request']==load(d/'capture/cases.json')
requests={}
for j,item in enumerate(data['items']):
 assert(item['repetition'],item['warmup'],item['policy'])==(j,j==0,'default')
 for phase,c in item['result']['model_response'].items():
  if not isinstance(c,dict)or c.get('status')!='ok':continue
  ri,ref=refs[(item['observation_id'],phase)];stable=lambda x:{k:v for k,v in x.items()if k not in ['model_response','model_response_json']};assert stable(item['result'])==stable(ri['result'])
  assert all(c[key]==ref[key]for key in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])and item['image_sha256']==ri['image_sha256']and item['skill_sha256']==ri['skill_sha256']
  req=c['timings']['counter_trace_request_id'];assert req>0 and req not in requests;count=c['generated_tokens'];limit=c['serving_configuration']['requested_output_limit'];steps=count-1 if count==limit else count
  requests[req]=dict(model=c['model_id'],warmup=item['warmup'],steps=[step for step in [4,16,28]if step<=steps])
assert len(requests)==4
windows=collections.defaultdict(list);commands={};pending=[]
for line in(d/'capture/candidate-1.counters.jsonl').read_text().splitlines():
 kind,payload=line.split(' ',1);c=json.loads(payload);assert c['request']in requests and c['model']==requests[c['request']]['model']and c['decode_step']in requests[c['request']]['steps']and c['diagnostic']and not c['serialized'];key=(c['request'],c['decode_step'])
 if kind=='metal_counter_profile':assert c['index']==len(pending)and 0<c['start_raw']<=c['end_raw']<2**64-1;pending.append(c);windows[key].append(c)
 else:
  assert kind=='metal_counter_command'and key not in commands and c['samples']==len(pending)>0 and c['samples']<=2048 and c['partial_submissions']==0
  assert all(all(row[field]==c[field]for field in ['pid','context','command','request','model','decode_step'])for row in pending);commands[key]=c;pending=[]
assert not pending and set(commands)=={(req,step)for req,row in requests.items()for step in row['steps']}and len(commands)==12 and sum(map(len,windows.values()))==7716
analysis=load(d/'capture/counter-analysis.json');assert len(analysis['windows'])==12
def close(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
aggregate=collections.defaultdict(lambda:collections.defaultdict(lambda:[0,0]))
for key,events in windows.items():
 req,step=key;totals=collections.defaultdict(lambda:[0,0]);intervals=[]
 for c in events:
  delta=c['end_raw']-c['start_raw'];totals[c['op']][0]+=1;totals[c['op']][1]+=delta;intervals.append((c['start_raw'],c['end_raw']))
 merged=[]
 for start,end in sorted(intervals):
  if merged and start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],end)
  else:merged.append([start,end])
 total=sum(v[1]for v in totals.values());union=sum(end-start for start,end in merged);span=max(end for _,end in intervals)-min(start for start,_ in intervals)
 row=next(row for row in analysis['windows']if row['request']==req and row['decode_step']==step)
 assert row['sum_interval_duration_raw']==total and row['union_interval_duration_raw']==union and row['overlap_raw']==total-union and row['span_between_first_last_sample_raw']==span and row['gap_within_sample_span_raw']==span-union
 assert row['overlap_raw']>0
 for rank in row['ranks']:
  assert [rank['passes'],rank['duration_raw']]==totals[rank['op']];close(rank['fraction_of_sampled_interval_duration_sum'],rank['duration_raw']/total)
 for op,value in totals.items():g=aggregate[(requests[req]['model'],requests[req]['warmup'])][op];g[0]+=value[0];g[1]+=value[1]
for row in analysis['first_and_warm_separate_aggregates']:
 totals=aggregate[(row['model'],row['warmup'])];total=sum(v[1]for v in totals.values())
 for rank in row['ranks']:assert [rank['passes'],rank['duration_raw']]==totals[rank['op']];close(rank['fraction_of_sampled_interval_duration_sum'],rank['duration_raw']/total)
calls=list(csv.DictReader((d/'calls.csv').open()));parents=list(csv.DictReader((d/'skill-requests.csv').open()));encoders=list(csv.DictReader((d/'capture/encoder-samples.csv').open()))
assert len(calls)==4 and len(parents)==2 and len(encoders)==7716 and all(row['eligible_for_speed']=='False'for rows in [calls,parents,encoders]for row in rows)
inputs=load(d/'frozen-input-identities.json');assert sha(d/inputs['frame_relative'])==inputs['image_sha256']and sha(d/inputs['serialized_skill_relative'])==inputs['recorded_skill_sha256']
assert load(d/inputs['serialized_skill_relative'])==data['items'][0]['skill']
for rel,info in load(d/'frozen-native-guard-identities.json')['files'].items():assert sha(d/rel)==info['sha256']==load(d/'learned-identities-before.json')['external'][info['original_path']]
assert load(d/'decision.json')['decision']=='diagnostic-complete'and not load(d/'decision.json')['promotion']
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_files=len(manifest),source_files=1720,overlays=len(recon['overlay_files']),metadata_delta=changed,original_words_states_tokens_goldens_scope_windows_rawranks_unions_overlap_and_CSV_recomputed=True,calls=4,requests=2,windows=12,samples=7716,no_speed_or_promotion_claim=True)
(Path(__file__).parent/'ws62-archive-independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
