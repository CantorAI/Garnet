from pathlib import Path
import json,hashlib,sys
w=Path(__file__).resolve().parent;p=w/'functional-smoke';sys.path.insert(0,str(w));from qos_call_guard import qos_valid
source=Path('/Users/shawn/CantorAI/out/research/worldsense-thread-qos/source');identity=json.loads((p/'before-functional-identities.json').read_text());assert all(hashlib.sha256((source/k).read_bytes()).hexdigest()==v for k,v in identity['source'].items());assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in identity['drivers'].items())
data=json.loads((p/'candidate-1.json').read_text());assert data['valid']and data['assignments_restored']and len(data['items'])==2 and not data['search_plans'];assert data['request']==json.loads((p/'cases.json').read_text())
gold=json.loads(Path('/Users/shawn/CantorAI/Garnet/results/20261016/worldsense-m5-prefix-cache/confirmation/paired-1.json').read_text());refs={(i['observation_id'],phase):(i,c)for i in gold['items']if i['policy']=='candidate'for phase,c in i['result']['model_response'].items()if isinstance(c,dict)and c.get('status')=='ok'};checks=[];child_identity=None;observed=[]
for j,item in enumerate(data['items']):
 assert(item['repetition'],item['warmup'],item['policy'])==(j,j==0,'default')
 for phase,c in item['result']['model_response'].items():
  if not isinstance(c,dict)or c.get('status')!='ok':continue
  ri,ref=refs[(item['observation_id'],phase)];stable=lambda x:{k:v for k,v in x.items()if k not in ['model_response','model_response_json']};assert stable(item['result'])==stable(ri['result']);assert all(c[k]==ref[k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration']);assert c['timings']['input_tokens']==ref['timings']['input_tokens']and item['image_sha256']==ri['image_sha256']and item['skill_sha256']==ri['skill_sha256']
  g=c['timings']['native_gateway'];parent=g['parent_thread_qos_requested'];assert parent['get_status']==parent['thread_id_status']==0 and parent['active_scopes']==parent['end_errors']==0
  memory=c['timings']['memory_after_decode'];parts=memory['process_samples'];assert memory['cohort_process_count']==2 and len(parts)==2 and len({x['pid']for x in parts})==2
  for key in ['metal_allocated_bytes','task_physical_footprint_bytes','task_resident_bytes','task_compressed_bytes','task_internal_bytes','task_virtual_bytes']:assert memory[key]==sum(x[key]for x in parts)
  child=g['child'];who=(child['support']['pid'],child['birth_id']);assert child_identity in [None,who];child_identity=who;assert child['inference_qos_mode']=='user_initiated'
  if c['model_id']=='Qwen3-1.7B':
   qos_valid(c);assert c['native_worker_call_index']==j+1 and child['infer_calls']==j+1;observed.append(dict(parent_requested_class=parent['requested_class'],child_requested_class=c['native_worker_qos']['before']['requested_class'],scope_target=c['native_worker_qos']['scope']['target_class'],before_thread=c['native_worker_qos']['before']['thread_id'],parent_thread=parent['thread_id'],scope_closed=c['native_worker_qos']['scope']['closed']))
  else:assert g['route']=='local'and c['image_preprocess_backend']=='native-cpp'
  checks.append(dict(model=c['model_id'],phase=phase,exact=True))
assert len(checks)==4 and len(observed)==2;assert json.loads((w/'restoration-verification.json').read_text())['benchmark_valid_request_http404']
(p/'functional-verification.json').write_text(json.dumps(dict(actualWorldSense_Galaxy_native_candidate_functional_pass=True,four_calls_exact=True,real_qos_scope_balanced_and_whole_process_memory_pass=True,observed_request_contexts=observed,checks=checks,source_drivers_unchanged=True,speed_eligibility=False,scope='Two fixed-frame Skill requests, candidate only, original outputs/inputs/budgets and actual scoped API/counter telemetry. No performance comparison or promotion.'),indent=2)+'\n');print('Actual WorldSense native QoS fourcall exact-output/identity/memory/scope functional smoke PASS; no speed claim')
