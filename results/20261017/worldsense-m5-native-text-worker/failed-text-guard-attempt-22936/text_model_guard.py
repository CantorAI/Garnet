"""Native RAW child full text state/cache/clock/memory correctness, not timing evidence."""
import sys,json,os,math,hashlib
from pathlib import Path
mode=sys.argv[1];w=Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-native-text-worker');s=Path('/Users/shawn/CantorAI/out/research/worldsense-native-text-worker/source');b=Path('/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-native-text-worker-'+mode);sys.path.insert(0,str(s));sys.path.insert(0,str(b))
import garnet
name='Qwen3-1.7B';model='/Users/shawn/CantorAI/out/models/'+name;profile=dict(maxInputTokens=192,maxOutputTokens=64,prefillChunk=16,textPrefixCache=True)
base='Decide whether the visual facts satisfy the required condition. Use only the observation as evidence. Missing or contradictory evidence means matched=false. Do not substitute a different subject or condition. A still image cannot prove a transition. Return only a compact JSON object with matched and reason. '*2
prompts=[base+'Observation: a red car passes a stationary viewpoint. Condition: a passenger vehicle is visible.',base+'Observation: people walk across a street. Condition: a passenger vehicle is visible.']
refs={r['prompt_index']:r for r in json.loads((w.parent/'worldsense-prefix-cache/prefix-model-guard.json').read_text())['records']};rows=[]
def info():return json.loads(garnet.native_text_worker_info_json(''))
def save(): (w/('text-model-progress-'+mode+'.json')).write_text(json.dumps(dict(records=rows),indent=2))
def check_memory(value):
 assert value['task_info_status']==0 and value['cohort_process_count']==2 and len(value['process_samples'])==2
 parts=value['process_samples'];assert {p['pid']for p in parts}=={os.getpid(),info()['children'][0]['support']['pid']}
 for key in ['metal_allocated_bytes','task_physical_footprint_bytes','task_resident_bytes','task_compressed_bytes','task_internal_bytes','task_virtual_bytes']:assert value[key]==sum(p[key]for p in parts)
 assert value['metal_recommended_working_set_bytes']==parts[0]['metal_recommended_working_set_bytes']and parts[0]['metal_registry_id']==parts[1]['metal_registry_id'];assert all(p['task_info_status']==0 for p in parts)
try:
 ready=json.loads(garnet.serve_model(model,str(s/'xModel/qwen/metal_dense'),'/Users/shawn/CantorAI/out/cache/'+name,json.dumps(profile),name));assert ready['ready'];initial=info();assert len(initial['children'])==1 and initial['children'][0]['models']==[name];birth=initial['children'][0]['birth_id']
 for j,index in enumerate([0,1,0]):
  call=json.loads(garnet.infer_json(prompts[index],'',8,name));assert call['status']=='ok';state=json.loads(garnet.native_text_worker_state_json(name));ref=refs[index];assert state['states']==ref['states']and call['generated_token_ids']==ref['call']['generated_token_ids'];assert len(state['states'])==28
  assert call['serving_configuration']==ref['call']['serving_configuration']and call['input_prompt_sha256']==ref['call']['input_prompt_sha256'];assert call['text']==ref['call']['text'];assert call['native_worker_call_index']==j+1
  t=call['timings'];g=t['native_gateway'];assert g['route']=='native_lrpc_raw_text'and g['child']['birth_id']==birth and g['child']['infer_calls']==j+1 and g['parent_pid']==os.getpid();assert 'native_text_worker'in g['parent_backend']and 'native_text_worker'not in call['backend'];assert call['serving_configuration']['text_prefix_cache']and t['prefix_cache_hit']==(j==2)
  assert g['gateway_started_monotonic']<=g['model_started_monotonic']<=g['first_token_monotonic'];assert abs(call['ttft_seconds']-(g['first_token_monotonic']-g['gateway_started_monotonic']))<1e-9;assert math.isfinite(call['total_seconds'])and call['total_seconds']>=call['ttft_seconds']
  assert g['raw_model_total_seconds']<=call['total_seconds']+.0001 and g['raw_model_ttft_seconds']<=call['ttft_seconds']+.0001;check_memory(t['memory_after_decode'])
  rows.append(dict(prompt_index=index,call=call,states=state['states']));save()
 error=json.loads(garnet.infer_json('word '*4096,'',8,name));assert error['status']=='error'and 'capacity'in error['error_message'];reset_state=json.loads(garnet.native_text_worker_state_json(name))['states'];assert info()['children'][0]['birth_id']==birth
 # Original Model.infer resets BEFORE checking capacity. Assert that original
 # reset semantics and subsequent exact recovery, not transactional rollback.
 for layer in reset_state:
  for part in layer:
   h=hashlib.sha256();remaining=part['values']
   while remaining:
    n=min(4096,remaining);h.update(bytes(n*4));remaining-=n
   assert part['sha256']==h.hexdigest()
 recovery=json.loads(garnet.infer_json(prompts[0],'',8,name));assert recovery['status']=='ok'and recovery['generated_token_ids']==refs[0]['call']['generated_token_ids']and json.loads(garnet.native_text_worker_state_json(name))['states']==refs[0]['states']
 # Explicitly close the native child model and reload with prefix disabled.
 assert json.loads(garnet.native_text_worker_close_json(name))['closed'];assert info()['children'][0]['models']==[]
 profile['textPrefixCache']=False;assert json.loads(garnet.serve_model(model,str(s/'xModel/qwen/metal_dense'),'/Users/shawn/CantorAI/out/cache/'+name,json.dumps(profile),name))['ready']
 call=json.loads(garnet.infer_json(prompts[0],'',8,name));assert call['status']=='ok'and not call['serving_configuration']['text_prefix_cache']and not call['timings']['prefix_cache_hit'];assert call['generated_token_ids']==refs[0]['call']['generated_token_ids']and json.loads(garnet.native_text_worker_state_json(name))['states']==refs[0]['states'];check_memory(call['timings']['memory_after_decode'])
 policy=json.loads(garnet.set_benchmark_prefill_policy(True));assert policy['supported']and policy['enabled'];again=json.loads(garnet.infer_json(prompts[0],'',8,name));assert again['status']=='ok'and again['serving_configuration']['text_prefix_cache']and again['generated_token_ids']==refs[0]['call']['generated_token_ids'];check_memory(again['timings']['memory_after_decode'])
 assert garnet.shutdown()and info()['children']==[]
 (w/('text-model-guard-'+mode+'.json')).write_text(json.dumps(dict(all28text_state_bits_tokens_prefix_capacity_error_profile_close_reload_policy_clock_cohort_memory_pass=True,first_three=rows,additional_prefix_disabled_call=call,policy_enabled_call=again,scope='Correctness only; no WorldSense speed/optimization claim.'),indent=2));print('Native child actual28text states/tokens/prefix/capacity/error/clock/process-set memory PASS '+mode,flush=True)
finally:garnet.shutdown()
