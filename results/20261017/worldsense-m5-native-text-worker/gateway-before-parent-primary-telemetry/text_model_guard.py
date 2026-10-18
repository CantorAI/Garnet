"""Native RAW child full text state/cache/clock/memory correctness, not timing evidence."""
import sys,json,os,math,hashlib,traceback
from pathlib import Path
mode=sys.argv[1];w=Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-native-text-worker');s=Path('/Users/shawn/CantorAI/out/research/worldsense-native-text-worker/source');b=Path('/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-native-text-worker-'+mode);sys.path.insert(0,str(s));sys.path.insert(0,str(b))
import garnet
name='Qwen3-1.7B';model='/Users/shawn/CantorAI/out/models/'+name;profile=dict(maxInputTokens=192,maxOutputTokens=64,prefillChunk=16,textPrefixCache=True)
base='Decide whether the visual facts satisfy the required condition. Use only the observation as evidence. Missing or contradictory evidence means matched=false. Do not substitute a different subject or condition. A still image cannot prove a transition. Return only a compact JSON object with matched and reason. '*2
prompts=[base+'Observation: a red car passes a stationary viewpoint. Condition: a passenger vehicle is visible.',base+'Observation: people walk across a street. Condition: a passenger vehicle is visible.']
prefix_path=w.parent/'worldsense-prefix-cache/prefix-model-guard.json';api_path=w.parent/'worldsense-isolated-primary/api-state-guard-off.json'
refs={r['prompt_index']:r for r in json.loads(prefix_path.read_text())['records']}
api_refs=[r for r in json.loads(api_path.read_text())['records']if r['model_id']==name]
assert len(api_refs)==3
for j,index in enumerate([0,1,0]):
 assert api_refs[j]['states']==refs[index]['states']and api_refs[j]['call']['generated_token_ids']==refs[index]['generated_token_ids']
 assert api_refs[j]['call']['input_prompt_sha256']==hashlib.sha256(prompts[index].encode()).hexdigest()==refs[index]['prompt_sha256']
rows=[];diagnostics=[]
def info():return json.loads(garnet.native_text_worker_info_json(''))
def save(): (w/('text-model-progress-'+mode+'.json')).write_text(json.dumps(dict(records=rows,diagnostics=diagnostics,oracle_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [prefix_path,api_path]}),indent=2))
def capture(label,operation):
 try:value=operation()
 except Exception:
  diagnostics.append(dict(case=label,exception=traceback.format_exc()));save();raise
 diagnostics.append(dict(case=label,value=value));save();return value

def infer(label,prompt):return capture(label,lambda:json.loads(garnet.infer_json(prompt,'',8,name)))
def states(label):return capture(label,lambda:json.loads(garnet.native_text_worker_state_json(name)))['states']
def check_memory(value):
 assert value['task_info_status']==0 and value['cohort_process_count']==2 and len(value['process_samples'])==2
 parts=value['process_samples'];assert {p['pid']for p in parts}=={os.getpid(),info()['children'][0]['support']['pid']}
 for key in ['metal_allocated_bytes','task_physical_footprint_bytes','task_resident_bytes','task_compressed_bytes','task_internal_bytes','task_virtual_bytes']:assert value[key]==sum(p[key]for p in parts)
 assert value['metal_recommended_working_set_bytes']==parts[0]['metal_recommended_working_set_bytes']and parts[0]['metal_registry_id']==parts[1]['metal_registry_id'];assert all(p['task_info_status']==0 for p in parts)
try:
 ready=capture('initial_serve',lambda:json.loads(garnet.serve_model(model,str(s/'xModel/qwen/metal_dense'),'/Users/shawn/CantorAI/out/cache/'+name,json.dumps(profile),name)));assert ready['ready'];initial=capture('initial_info',info);assert len(initial['children'])==1 and initial['children'][0]['models']==[name];birth=initial['children'][0]['birth_id']
 for j,index in enumerate([0,1,0]):
  call=infer('initial_infer_'+str(j),prompts[index]);assert call['status']=='ok';state=states('initial_states_'+str(j));ref=refs[index];expected=api_refs[j]['call'];rows.append(dict(prompt_index=index,call=call,states=state));save()
  assert state==ref['states']and call['generated_token_ids']==ref['generated_token_ids'];assert len(state)==28
  assert call['serving_configuration']==expected['serving_configuration']and call['input_prompt_sha256']==expected['input_prompt_sha256'];assert call['text']==expected['text'];assert call['native_worker_call_index']==j+1
  t=call['timings'];g=t['native_gateway'];assert g['route']=='native_lrpc_raw_text'and g['child']['birth_id']==birth and g['child']['infer_calls']==j+1 and g['parent_pid']==os.getpid();assert 'native_text_worker'in g['parent_backend']and 'native_text_worker'not in call['backend'];assert call['serving_configuration']['text_prefix_cache']and t['prefix_cache_hit']==expected['timings']['prefix_cache_hit']
  assert g['gateway_started_monotonic']<=g['model_started_monotonic']<=g['first_token_monotonic'];assert abs(call['ttft_seconds']-(g['first_token_monotonic']-g['gateway_started_monotonic']))<1e-9;assert math.isfinite(call['total_seconds'])and call['total_seconds']>=call['ttft_seconds']
  assert g['raw_model_total_seconds']<=call['total_seconds']+.0001 and g['raw_model_ttft_seconds']<=call['ttft_seconds']+.0001;check_memory(t['memory_after_decode'])
 error=infer('capacity_error','word '*4096);assert error['status']=='error'and 'capacity'in error['error_message'];reset_state=states('capacity_error_reset_states');assert info()['children'][0]['birth_id']==birth
 # Original Model.infer resets BEFORE checking capacity. Assert original reset
 # semantics and subsequent exact recovery, not transactional rollback.
 for layer in reset_state:
  for part in layer:
   h=hashlib.sha256();remaining=part['values']
   while remaining:
    n=min(4096,remaining);h.update(bytes(n*4));remaining-=n
   assert part['sha256']==h.hexdigest()
 recovery=infer('capacity_recovery',prompts[0]);recovery_state=states('capacity_recovery_states');assert recovery['status']=='ok'and recovery['generated_token_ids']==refs[0]['generated_token_ids']and recovery_state==refs[0]['states']
 assert capture('close',lambda:json.loads(garnet.native_text_worker_close_json(name)))['closed'];assert info()['children'][0]['models']==[]
 profile['textPrefixCache']=False;assert capture('reload_prefix_disabled',lambda:json.loads(garnet.serve_model(model,str(s/'xModel/qwen/metal_dense'),'/Users/shawn/CantorAI/out/cache/'+name,json.dumps(profile),name)))['ready']
 call=infer('prefix_disabled',prompts[0]);state=states('prefix_disabled_states');assert call['status']=='ok'and not call['serving_configuration']['text_prefix_cache']and not call['timings']['prefix_cache_hit'];assert call['generated_token_ids']==refs[0]['generated_token_ids']and state==refs[0]['states'];check_memory(call['timings']['memory_after_decode'])
 policy=capture('enable_policy',lambda:json.loads(garnet.set_benchmark_prefill_policy(True)));assert policy['supported']and policy['enabled'];again=infer('policy_enabled',prompts[0]);assert again['status']=='ok'and again['serving_configuration']['text_prefix_cache']and again['generated_token_ids']==refs[0]['generated_token_ids'];check_memory(again['timings']['memory_after_decode'])
 assert garnet.shutdown()and info()['children']==[]
 (w/('text-model-guard-'+mode+'.json')).write_text(json.dumps(dict(all28text_state_bits_tokens_prefix_capacity_error_profile_close_reload_policy_clock_cohort_memory_pass=True,first_three=rows,additional_prefix_disabled_call=call,policy_enabled_call=again,scope='Correctness only; no WorldSense speed/optimization claim.'),indent=2));print('Native child actual28text states/tokens/prefix/capacity/error/clock/process-set memory PASS '+mode,flush=True)
except Exception:
 diagnostics.append(dict(case='guard_failure',exception=traceback.format_exc()));save();raise
finally:garnet.shutdown()
