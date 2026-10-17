from pathlib import Path
import json,hashlib,csv
w=Path(__file__).resolve().parent;repo=next((parent for parent in w.parents if (parent/'results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json').is_file()),Path('/Users/shawn/CantorAI/Garnet'));p=json.loads((w/'inventory-1.json').read_text());g=json.loads((repo/'results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json').read_text());assert p['valid']and p['assignments_restored']and len(p['items'])==6 and len(p['search_plans'])==2
ref={};qref={}
for i in g['items']:
 if i['policy']!='candidate':continue
 for phase,c in i['result'].get('model_response',{}).items():
  if isinstance(c,dict)and c.get('status')=='ok':ref[(i['observation_id'],phase)]=(i,c)
for i in g['search_plans']:
 if i['policy']=='candidate':qref[i['query']]=i['result']['benchmark_model_response']
calls=[]
for repetition in [0,1]:
 for item in p['items']:
  if item['repetition']!=repetition:continue
  for phase,c in item['result'].get('model_response',{}).items():
   if not isinstance(c,dict)or c.get('status')!='ok':continue
   ri,rc=ref[(item['observation_id'],phase)]
   assert item['image_sha256']==ri['image_sha256']and item['skill_sha256']==ri['skill_sha256']
   calls.append((dict(window=repetition,observation_id=item['observation_id'],phase=phase,query=''),c,rc))
 for item in p['search_plans']:
  if item['repetition']==repetition:calls.append((dict(window=repetition,observation_id='',phase='search_plan',query=item['query']),item['result']['benchmark_model_response'],qref[item['query']]))
assert len(calls)==12
snapshots=[];initialization={}
for index,(meta,c,rc)in enumerate(calls):
 assert c['generated_token_ids']==rc['generated_token_ids']and c['text']==rc['text']and c['input_prompt_sha256']==rc['input_prompt_sha256']and c['serving_configuration']==rc['serving_configuration']
 assert c['timings']['input_tokens']==rc['timings']['input_tokens']
 initialization[c['model_id']]=c['timings']['initialization_memory']
 for phase in ['memory_before_request','memory_after_input','memory_after_prefill','memory_after_decode']:
  s=c['timings'][phase];assert s['task_info_status']==0 and s['metal_recommended_working_set_bytes']>0
  snapshots.append(dict(call_index=index,model_id=c['model_id'],snapshot_phase=phase,**meta,**s))
assert len({s['metal_registry_id']for s in snapshots})==1
manifest=json.loads((w/'starting-source-manifest.json').read_text());source=Path('/Users/shawn/CantorAI/out/research/worldsense-inventory/source');source_audit='live_full_manifest' if source.is_dir() else 'archived_subset_requires_compatible_base';assert all(hashlib.sha256((source/rel).read_bytes()).hexdigest()==h for rel,h in manifest.items()) if source.is_dir() else all(hashlib.sha256(f.read_bytes()).hexdigest()==manifest[str(f.relative_to(w/'source'))]for f in (w/'source').rglob('*')if f.is_file())
init=[x[phase]for x in initialization.values()for phase in ['before','after_decoder_ready']]
for s in init:assert s['task_info_status']==0
summary=dict(source_audit=source_audit,scope='Diagnostic12-call WorldSense execution, no speed/memory improvement claim; maxima are sampled, not peaks. Metal/task metrics overlap and must not be added.',all12_full_outputs_prompts_configs_and_inputs_equal_to_WS10_confirmation=True,initialization_memory=initialization,snapshot_count=len(snapshots),sampled_max_metal_bytes=max(s['metal_allocated_bytes']for s in snapshots),recommended_working_set_bytes=snapshots[0]['metal_recommended_working_set_bytes'],sampled_max_physical_footprint_bytes=max(s['task_physical_footprint_bytes']for s in snapshots),sampled_max_task_compressed_bytes=max(s['task_compressed_bytes']for s in snapshots),window_last_metal_bytes={str(i):[s['metal_allocated_bytes']for s in snapshots if s['window']==i][-1]for i in [0,1]},snapshots=snapshots)
(w/'analysis.json').write_text(json.dumps(summary,indent=2))
with (w/'memory-snapshots.csv').open('w',newline='')as f:
 writer=csv.DictWriter(f,fieldnames=list(snapshots[0]));writer.writeheader();writer.writerows(snapshots)
print(json.dumps({k:v for k,v in summary.items()if k not in ['snapshots','initialization_memory']},indent=2))
