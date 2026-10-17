"""Real native serving boundary correctness only. Timings are not speed evidence."""
import sys,pathlib,json,hashlib,struct,math
mode=sys.argv[1];assert mode in ['off','on'];expected=1 if mode=='on'else 0
source='/Users/shawn/CantorAI/out/research/worldsense-resident-admission/source';build='/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-resident-admission-'+mode
sys.path.insert(0,build);sys.path.insert(0,source);sys.path.insert(0,source+'/runtime')
import garnet,metal_serving
w=pathlib.Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-resident-admission');records=[];loads=[]
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
 call=json.loads(garnet.infer_json(prompt,image,limit,name));(w/('last-native-call-'+mode+'.json')).write_text(json.dumps(call,indent=2));assert call['status']=='ok'
 timings=call['timings'];assert (timings['resident_avoided_storage_bytes']>0)==bool(expected);assert (timings['lazy_default_planned_slots']>0)==bool(expected);assert timings['lazy_default_rehydrated_bytes']==0 and timings['lazy_default_retired_bytes']==0
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
(w/('api-state-guard-'+mode+'.json')).write_text(json.dumps(dict(real_native_api_boundaries=True,all32_VLM_and28_text_state_bits_and_tokens_match_previous=True,capacity_rejected_and_recovered=True,lazy_defaults_expected=expected,records=records,loads=loads),indent=2))
print('Real native load/infer boundaries,32VLM/28text finite states and tokens, cold/warm prefix and capacity recovery PASS '+mode,flush=True)
