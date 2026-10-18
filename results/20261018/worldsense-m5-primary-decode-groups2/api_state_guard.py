"""Real native serving boundary correctness only. Timings are not speed evidence."""
import sys,pathlib,json,hashlib,struct,math
mode=sys.argv[1];assert mode in ['off','primary','on','compat56'];expected=0 if mode=='off'else 1;vector_expected=mode in ['on','compat56'];core_expected=False;compat_expected=mode=='compat56';groups2_expected=mode=='on'
source='/Users/shawn/CantorAI/out/research/worldsense-primary-decode-groups2/source';build='/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-primary-decode-groups2-'+mode
sys.path.insert(0,build);sys.path.insert(0,source);sys.path.insert(0,source+'/runtime')
import garnet,metal_serving
w=pathlib.Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-primary-decode-groups2');records=[];loads=[]
def save(): (w/('api-state-progress-'+mode+'.json')).write_text(json.dumps(dict(loads=loads,records=records),indent=2))
def states(name,layers):
 m=metal_serving._models[name];result=[]
 for layer in range(layers):
  parts=[]
  for tensor in m.decoder.session.snapshot(layer):
   values=tensor.tolist();assert all(math.isfinite(v)for v in values);h=hashlib.sha256()
   for i in range(0,len(values),4096):
    b=values[i:i+4096];h.update(struct.pack('<'+'f'*len(b),*b))
   parts.append(dict(values=len(values),sha256=h.hexdigest()))
  result.append(parts)
 return result
def load(name,xmodel,profile):
 result=json.loads(garnet.serve_model('/Users/shawn/CantorAI/out/models/'+name,source+'/'+xmodel,'/Users/shawn/CantorAI/out/cache/'+name,json.dumps(profile),name));loads.append(result);save();assert result['ready'];assert name in metal_serving._models

def infer(name,prompt,image,limit,layers):
 call=json.loads(garnet.infer_json(prompt,image,limit,name));assert ('primary_vector_prefill' in call.get('backend',''))==vector_expected;assert ('primary_prefill_isolated_library' in call.get('backend',''))==vector_expected;assert ('primary_bf16_blocks' in call.get('backend',''))==bool(expected);assert ('primary_vector_decode' in call.get('backend',''))==bool(expected);assert ('primary_kmajor_decode' in call.get('backend',''))==bool(expected);assert ('primary_u32_metadata' in call.get('backend',''))==bool(expected);(w/('last-native-call-'+mode+'.json')).write_text(json.dumps(call,indent=2));assert call['status']=='ok'
 assert ('primary_full_isolated_library'in call['backend'])==vector_expected
 assert ('request_scope_weight_residency'in call['backend'])==compat_expected
 timings=call['timings'];assert (timings['lazy_default_retired_bytes']>0)==bool(expected) and (timings['lazy_default_planned_slots']>0)==bool(expected) and timings['lazy_default_rehydrated_bytes']==0
 assert ('lazy_resident_defaults'in call['backend'])==bool(expected) and ('cached_matrix_metadata'in call['backend'])==compat_expected
 stats=timings['primary_storage_after'];delta=timings['primary_encoded_delta']
 graphs=[x['graph']for x in metal_serving._models[name].decoder.layers]+list(metal_serving._models[name].decoder.prefill_graphs.values())+[metal_serving._models[name].decoder.head]
 assert all(bool(g.wide_primary_eligible)==(bool(expected)and name=='Qwen3.5-4B')for g in graphs)
 assert ('wide_primary_storage'in call['backend'])==bool(expected)
 if expected and name=='Qwen3.5-4B':
  assert stats['active_weights']>0 and stats['original_bytes']>stats['payload_bytes']+stats['metadata_bytes']>0 and stats['allocated_bytes']>0
  assert all(delta[k]>0 for k in ['decode_encoded','prefill_encoded','head_encoded'])
  assert delta['vector_decode_encoded']==delta['decode_encoded'] and delta['vector_head_encoded']==delta['head_encoded']
 elif expected:
  assert all(stats[k]==0 for k in ['active_weights','original_bytes','payload_bytes','metadata_bytes','allocated_bytes'])and all(v==0 for v in delta.values())
 else:assert all(v==0 for v in stats.values()) and all(v==0 for v in delta.values())

 if compat_expected:
  q=timings['request_weight_residency'];assert q['enabled']and not q['scope_open']and q['active_bytes']==q['global_used_bytes']==0 and q['calls']==q['ends']and q['limit_bytes']==4294967296
  if name=='Qwen3-1.7B':assert q['reason']==1 and q['last_count']==197 and q['last_bytes']==3440902144 and q['coverage_cache_hits']>0 and q['coverage_fallbacks']==0 and 0<q['last_coverage_graphs']<=256
  else:assert q['reason']==7 and q['admitted']==0 and q['fallback']==q['calls']
 else:assert 'request_weight_residency'not in timings
 assert ('primary_decode_groups2'in call['backend'])==groups2_expected
 for key,oldkey in [('decode_groups2_encoded','decode_encoded'),('head_groups2_encoded','head_encoded')]:
  before_group=timings['primary_storage_before'][key];after_group=timings['primary_storage_after'][key]
  assert type(before_group)is int and type(after_group)is int and 0<=before_group<=after_group
  assert after_group-before_group==(delta[oldkey]if groups2_expected else 0)
  if not groups2_expected:assert before_group==after_group==0
  if name=='Qwen3-1.7B':assert before_group==after_group
 assert ('primary_core_isolated_library'in call['backend'])==core_expected
 before=timings['primary_storage_before'];after=timings['primary_storage_after']
 for key,oldkey in [('core_library_decode_encoded','decode_encoded'),('core_library_prefill_encoded','prefill_encoded'),('core_library_head_encoded','head_encoded')]:
  assert type(before[key])is int and type(after[key])is int and 0<=before[key]<=after[key]
  core_delta=after[key]-before[key]
  assert core_delta==(delta[oldkey]if core_expected else 0)
  if not core_expected:assert before[key]==after[key]==0
  if name=='Qwen3-1.7B':assert core_delta==0

 row=dict(model_id=name,call=call,states=states(name,layers));records.append(row);save();return row
load('Qwen3.5-4B','xModel/qwen35/vl_4b',dict(maxInputTokens=384,maxOutputTokens=128,prefillChunk=64,linearChunk=True,imagePixels=65536))
prompt='Describe the visible subject in one sentence.';image='/Users/shawn/Documents/Codex/2026-10-15/i/work/skill-expansion/car.jpg'
a=infer('Qwen3.5-4B',prompt,image,16,32)
ref=json.loads((w.parent/'worldsense-prefix-cache/state-guard.json').read_text())['records'][1]
assert a['states']==ref['states']and a['call']['generated_token_ids']==ref['generated_token_ids']
bad=json.loads(garnet.infer_json('word '*4096,'',16,'Qwen3.5-4B'));assert bad['status']=='error'and 'capacity'in bad['error_message']
b=infer('Qwen3.5-4B',prompt,image,16,32);assert a['states']==b['states']and a['call']['generated_token_ids']==b['call']['generated_token_ids']
# Release explicit owners through their normal APIs before loading the companion.
metal_serving._models['Qwen3.5-4B'].decoder.close();metal_serving._models['Qwen3.5-4B'].vision.close();del metal_serving._models['Qwen3.5-4B']
load('Qwen3-1.7B','xModel/qwen/metal_dense',dict(maxInputTokens=192,maxOutputTokens=64,prefillChunk=16,textPrefixCache=True))
base='Decide whether the visual facts satisfy the required condition. Use only the observation as evidence. Missing or contradictory evidence means matched=false. Do not substitute a different subject or condition. A still image cannot prove a transition. Return only a compact JSON object with matched and reason. '*2
prompts=[base+'Observation: a red car passes a stationary viewpoint. Condition: a passenger vehicle is visible.',base+'Observation: people walk across a street. Condition: a passenger vehicle is visible.']
previous=json.loads((w.parent/'worldsense-prefix-cache/prefix-model-guard.json').read_text())['records'];references={r['prompt_index']:r for r in previous}
for i in [0,1,0]:
 row=infer('Qwen3-1.7B',prompts[i],'',8,28);ref=references[i]
 assert row['states']==ref['states']and row['call']['generated_token_ids']==ref['generated_token_ids']
 assert row['call']['timings']['prefix_cache_hit']==(len(records)>3)
(w/('api-state-guard-'+mode+'.json')).write_text(json.dumps(dict(real_native_api_boundaries=True,all32_VLM_and28_text_state_bits_and_tokens_match_previous=True,capacity_rejected_and_recovered=True,primary_expected=expected,vector_prefill_expected=vector_expected,lazy_defaults_expected=bool(expected),cached_metadata_expected=compat_expected,records=records,loads=loads),indent=2))
print('Real native load/infer boundaries,32VLM/28text finite states and tokens, cold/warm prefix and capacity recovery PASS '+mode,flush=True)
