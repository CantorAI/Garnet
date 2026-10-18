"""Native prefix session safety and recovery, correctness only; not a benchmark."""
import sys,pathlib,json,hashlib,struct,math
build='/Users/shawn/CantorAI/out/build/garnet-main-macos-verify-20261018';source='/Users/shawn/CantorAI/Garnet';sys.path.insert(0,build);sys.path.insert(0,source)
import garnet,tensor
from runtime.metal_serving import Model
from xModel.qwen.metal_dense.layers import DenseLayer,T
w=pathlib.Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/garnet-main-merge-20261018');reference=json.loads((w/'prefix-model-guard.json').read_text())['records'][0]
base='Decide whether the visual facts satisfy the required condition. Use only the observation as evidence. Missing or contradictory evidence means matched=false. Do not substitute a different subject or condition. A still image cannot prove a transition. Return only a compact JSON object with matched and reason. '*2
prompt=base+'Observation: a red car passes a stationary viewpoint. Condition: a passenger vehicle is visible.'
m=Model('/Users/shawn/CantorAI/out/models/Qwen3-1.7B',source+'/xModel/qwen/metal_dense','/Users/shawn/CantorAI/out/cache/Qwen3-1.7B',{'maxInputTokens':192,'maxOutputTokens':64,'prefillChunk':16,'textPrefixCache':True},'Qwen3-1.7B');checks=[]
def check(name,ok):
 assert ok,name;checks.append(name);(w/'prefix-safety-progress.json').write_text(json.dumps({'passed_checks':checks},indent=2))
def reject(name,fn):
 rejected=False
 try:fn()
 except Exception:rejected=True
 check(name,rejected)
def first_state():return [v.tolist()for v in m.decoder.session.snapshot(0)]
def zero():return all(v==0 for values in first_state()for v in values)
def all_states():
 result=[]
 for layer in range(28):
  parts=[]
  for v in m.decoder.session.snapshot(layer):
   values=v.tolist();assert all(math.isfinite(x)for x in values);h=hashlib.sha256()
   for i in range(0,len(values),4096):
    part=values[i:i+4096];h.update(struct.pack('<'+'f'*len(part),*part))
   parts.append({'values':len(values),'sha256':h.hexdigest()})
  result.append(parts)
 return result
other=garnet.decoder_session()
try:
 call=m.infer(prompt,'',8);check('identical_reference_prompt_and_tokens',call['input_prompt_sha256']==reference['prompt_sha256']and call['generated_token_ids']==reference['generated_token_ids']);session=m.decoder.session;handle=next(iter(m._prefix_keys.values()));saved=first_state()
 for count in [0,63,65,-1]:reject('invalid_capture_length_'+str(count),lambda:session.capture_prefix(count))
 check('invalid_capture_did_not_mutate',first_state()==saved)
 reject('restore_without_reset_rejected',lambda:session.restore_prefix(handle));check('failed_restore_did_not_mutate',first_state()==saved)
 session.reset();reject('unknown_handle_rejected',lambda:session.restore_prefix(handle+1000000));check('unknown_handle_left_zero_state',zero());check('valid_handle_restores_after_rejection',session.restore_prefix(handle)==64)
 graphs=[layer['graph']for layer in m.decoder.layers];other.build(graphs,m.decoder.head);reject('foreign_session_handle_rejected',lambda:other.restore_prefix(handle));check('other_session_cache_empty',other.prefix_cache_entries==0 and other.prefix_cache_bytes==0)
 for family in ['Different family B: ','Different family C: ']:m.infer(family+prompt,'',8)
 expected=2*28*m.decoder.config['num_key_value_heads']*64*m.decoder.config['head_dim']*4
 check('two_entry_payload_bound',session.prefix_cache_entries==2 and session.prefix_cache_bytes==2*expected and session.prefix_cache_bytes<=64*1024*1024)
 session.reset();reject('evicted_handle_rejected',lambda:session.restore_prefix(handle));check('eviction_rejection_left_zero_state',zero())
 recovered=m.infer(prompt,'',8);check('evicted_prefix_recomputed',not recovered['timings']['prefix_cache_hit']);check('eviction_recovery_all_state_bits',all_states()==reference['states']and recovered['generated_token_ids']==reference['generated_token_ids'])
 old=list(m._prefix_keys.values())[-1];session.clear_prefix_cache();m._prefix_keys.clear();check('explicit_clear_releases_payload',session.prefix_cache_entries==0 and session.prefix_cache_bytes==0);session.reset();reject('cleared_handle_rejected',lambda:session.restore_prefix(old))
 m.infer(prompt,'',8);old=next(iter(m._prefix_keys.values()));session.build(graphs,m.decoder.head);check('rebuild_clears_cache',session.prefix_cache_entries==0 and session.prefix_cache_bytes==0);reject('rebuild_invalidates_old_handle',lambda:session.restore_prefix(old));m._prefix_keys.clear()
 recovered=m.infer(prompt,'',8);handle=next(iter(m._prefix_keys.values()));check('rebuild_recovery_tokens',recovered['generated_token_ids']==reference['generated_token_ids'])
 # A nonfinite computation without prediction must never become a cached prefix.
 m.decoder.reset();pg=m.decoder._resident_prefill_graphs(16)
 hidden=tensor.tensor([float('nan')]*(16*2048),shape=[1,16,2048],dtype=tensor.float32);positions=tensor.tensor([v for i in range(16)for v in [i,i,i]],shape=[1,16,3],dtype=tensor.float32);slots=tensor.tensor(list(range(16)),shape=[1,16],dtype=tensor.float32);flags=tensor.tensor([1]*16,shape=[1,16],dtype=tensor.float32)
 session.run(pg,hidden,positions,slots,flags,-1);before_count=session.prefix_cache_entries;reject('nonfinite_state_not_cached',lambda:session.capture_prefix(64));check('failed_capture_preserves_entries',session.prefix_cache_entries==before_count)
 recovered=m.infer(prompt,'',8);check('nonfinite_failure_full_recovery',recovered['generated_token_ids']==reference['generated_token_ids']and all_states()==reference['states'])
 # Same graph object, new generation: reject before restoring any private state.
 m.decoder.reset();graph=graphs[0];graph.close();reject('closed_owner_rejected',lambda:session.restore_prefix(handle));check('closed_owner_rejection_left_zero_state',zero())
 c=m.decoder.config;capacity=m.capacity;shapes=m.decoder.layers[0]['shapes'];symbols=[tensor.input('prefix_generation_'+str(i),shape=shape,dtype=tensor.float32)for i,shape in enumerate(shapes)]
 @T.fusion(name='prefix_rebuilt_owner',boundary='required')
 def forward(hidden,key,value,positions,slots,active):return DenseLayer(hidden,key,value,positions,slots,active,c,0,capacity)
 graph.build_bf16(forward(*symbols),symbols,m.checkpoint);reject('rebuilt_owner_generation_rejected',lambda:session.restore_prefix(handle));check('rebuilt_owner_rejection_left_zero_state',zero())
 session.close();reject('closed_session_rejected',lambda:session.restore_prefix(handle));check('closed_session_releases_prefix_payload',session.prefix_cache_entries==0 and session.prefix_cache_bytes==0)
 (w/'prefix-safety-guard.json').write_text(json.dumps({'scope':'Native session/cache correctness only; all timings excluded from performance evidence','passed':True,'checks':checks,'logical_single_entry_bytes':expected,'logical_max_observed_bytes':2*expected,'reference_all_layer_states':reference['states']},indent=2));print('Native prefix safety/eviction/session ownership/owner generation/nonfinite/reset/recovery PASS',len(checks),'checks',flush=True)
finally:other.close();m.decoder.close()
