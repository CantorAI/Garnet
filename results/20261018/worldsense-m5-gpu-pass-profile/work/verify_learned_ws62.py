"""Independent frozen-input/model parity and explicit GPU-counter scope audit."""
from pathlib import Path
import json,hashlib,subprocess,sys,datetime
w=Path(__file__).resolve().parent.parent/'worldsense-gpu-counter-profile';r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-gpu-counter-profile/source'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
frozen=json.loads((w/'learned-identities-before.json').read_text());assert all(sha(s/k)==v for k,v in frozen['source'].items())and all(sha(Path(k))==v for k,v in frozen['external'].items())
outer=json.loads((w/'outer-controller-identity-observation.json').read_text());assert sha(Path(outer['path']))==outer['sha256']
assert json.loads((w/'learned-controller-result.json').read_text())['full_original_learned_numerical_and_counter_fault_model_proof_pass']
subprocess.run([sys.executable,str(w/'verify_correctness_terminal.py'),sys.argv[1]],check=True)
scope={}
for mode in ['off','primary','on','quiet']:
 records=json.loads((w/('api-state-progress-'+mode+'.json')).read_text())['records'];assert len(records)==5
 path=w/('api-state-guard-'+mode+'.log.counters.jsonl')
 if mode=='quiet':
  assert not path.exists()and all('counter_trace_request_id'not in row['call']['timings']for row in records);scope[mode]=dict(defaultOFF_no_trace_or_request_metadata=True);continue
 samples=[];commands=[];pending=[]
 for line in path.read_text().splitlines():
  kind,payload=line.split(' ',1);value=json.loads(payload);assert value['diagnostic']and not value['serialized']and value['decode_step']in [4,16,28]
  if kind=='metal_counter_profile':
   assert value['index']==len(pending)and 0<value['start_raw']<=value['end_raw']<2**64-1;pending.append(value);samples.append(value)
  else:
   assert kind=='metal_counter_command'and value['samples']==len(pending)>0 and value['samples']<=2048 and value['partial_submissions']==0
   assert all(all(row[key]==value[key]for key in ['pid','context','command','request','model','decode_step'])for row in pending);commands.append(value);pending=[]
 assert not pending
 expected={}
 for row in records:
  c=row['call'];req=c['timings']['counter_trace_request_id'];assert isinstance(req,int)and req>0 and req not in expected
  count=c['generated_tokens'];limit=c['serving_configuration']['requested_output_limit'];steps=count-1 if count==limit else count
  expected[req]=dict(model=c['model_id'],steps=[step for step in [4,16,28]if step<=steps])
 assert len(commands)==sum(len(x['steps'])for x in expected.values())
 assert sorted((c['request'],c['model'],c['decode_step'])for c in commands)==sorted((req,data['model'],step)for req,data in expected.items()for step in data['steps'])
 scope[mode]=dict(calls=5,exact_request_model_step_joins=True,samples=len(samples),commands=len(commands),no_unsampled_phase_or_capacity_error_request_records=True)
recovery=json.loads((w/'counter-fault-model-recovery.json').read_text());assert recovery['actual_learned_counter_resolution_error_propagates_and_scope_clears']and recovery['all28_states_and_tokens_exact_after_reset_recovery']
(w/'learned-independent-audit.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observed_terminal_session=int(sys.argv[1]),observed_exit_code=0,source_runtime_models_binaries_and_actual_guard_identities_unchanged=True,outer_wrapper_sha_first_observed_after_launch_explicitly_disclosed=True,original_model_state_token_word_vision_verifier_pass=True,counter_request_scope_and_selected_steps=scope,actual_counter_error_recovery_pass=True,scope='Correctness and explicit counter identity/coverage only. No speed or optimized deployment claim.'),indent=2)+'\n')
print('WS62 full learned original guards and hardware-counter request/model/selected-step identities PASS')
