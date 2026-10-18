"""Actual learned-model poison/reset recovery after native counter resolution failure."""
from pathlib import Path
import sys,json
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-gpu-counter-profile/source';b=r/'out/build/garnet-metal-worldsense-gpu-counter-profile-fault';sys.path.insert(0,str(s));sys.path.insert(0,str(b));import garnet
assert hasattr(garnet,'_test_counter_emit_failure')and garnet.counter_trace_request_id()==0
name='Qwen3-1.7B';profile=dict(maxInputTokens=192,maxOutputTokens=64,prefillChunk=16,textPrefixCache=True)
base='Decide whether the visual facts satisfy the required condition. Use only the observation as evidence. Missing or contradictory evidence means matched=false. Do not substitute a different subject or condition. A still image cannot prove a transition. Return only a compact JSON object with matched and reason. '*2
prompt=base+'Observation: a red car passes a stationary viewpoint. Condition: a passenger vehicle is visible.'
result={}
def save():(w/'counter-fault-model-recovery.json').write_text(json.dumps(result,indent=2)+'\n')
def states():
 import metal_serving,hashlib,struct,math
 model=metal_serving._models[name];rows=[]
 for i in range(28):
  parts=[]
  for tensor in model.decoder.session.snapshot(i):
   values=tensor.tolist();assert all(math.isfinite(v)for v in values);h=hashlib.sha256()
   for start in range(0,len(values),4096):block=values[start:start+4096];h.update(struct.pack('<'+'f'*len(block),*block))
   parts.append(dict(values=len(values),sha256=h.hexdigest()))
  rows.append(parts)
 return rows
try:
 result['serve']=json.loads(garnet.serve_model(str(r/'out/models'/name),str(s/'xModel/qwen/metal_dense'),str(r/'out/cache'/name),json.dumps(profile),name));save();assert result['serve']['ready']
 assert garnet._test_counter_emit_failure(1)
 result['injected_call']=json.loads(garnet.infer_json(prompt,'',8,name));save();assert result['injected_call']['status']=='error'and 'Injected GPU trace resolution failure'in result['injected_call']['error_message']
 assert garnet.counter_trace_request_id()==0
 assert garnet._test_counter_emit_failure(0)
 result['recovery_call']=json.loads(garnet.infer_json(prompt,'',8,name));save();assert result['recovery_call']['status']=='ok'and garnet.counter_trace_request_id()==0
 result['states']=states();save()
 ref=next(row for row in json.loads((w.parent/'worldsense-isolated-primary/api-state-guard-off.json').read_text())['records']if row['model_id']==name)
 assert result['states']==ref['states']and all(result['recovery_call'][key]==ref['call'][key]for key in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])
 result.update(actual_learned_counter_resolution_error_propagates_and_scope_clears=True,all28_states_and_tokens_exact_after_reset_recovery=True,scope='Fault-only diagnostic correctness. No speed/optimization claim.');save();print('Actual learned counter failure/reset recovery,28 states/tokens and scope cleanup PASS',flush=True)
finally:garnet._test_counter_emit_failure(0);garnet.shutdown()
