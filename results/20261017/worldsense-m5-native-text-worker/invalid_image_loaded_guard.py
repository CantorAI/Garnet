"""Native loaded-model regression for actual Galaxy C++ absent-image and equivalents."""
import sys,json,hashlib
from pathlib import Path
w=Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-native-text-worker');s=Path('/Users/shawn/CantorAI/out/research/worldsense-native-text-worker/source');b=Path('/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-native-text-worker-fault');sys.path.insert(0,str(s));sys.path.insert(0,str(b));import garnet
name='Qwen3-1.7B';profile=dict(maxInputTokens=192,maxOutputTokens=64,prefillChunk=16,textPrefixCache=True)
base='Decide whether the visual facts satisfy the required condition. Use only the observation as evidence. Missing or contradictory evidence means matched=false. Do not substitute a different subject or condition. A still image cannot prove a transition. Return only a compact JSON object with matched and reason. '*2;prompt=base+'Observation: a red car passes a stationary viewpoint. Condition: a passenger vehicle is visible.'
ref=next(r for r in json.loads((w.parent/'worldsense-isolated-primary/api-state-guard-off.json').read_text())['records']if r['model_id']==name);result=dict(records=[])
def save():(w/'invalid-image-loaded-regression.json').write_text(json.dumps(result,indent=2)+'\n')
try:
 result['serve']=json.loads(garnet.serve_model('/Users/shawn/CantorAI/out/models/'+name,str(s/'xModel/qwen/metal_dense'),'/Users/shawn/CantorAI/out/cache/'+name,json.dumps(profile),name));save();assert result['serve']['ready'];initial=json.loads(garnet.native_text_worker_info_json(''))['children'][0];assert initial['infer_calls']==0
 for i,pair in enumerate([('cpp_default_invalid',None),('native_None',None),('native_number',17),('native_dict',{}),('native_empty_bytes',b''),('original_empty_string','')]):
  kind,image=pair
  if kind=='cpp_default_invalid':call=json.loads(garnet.native_invalid_image_test_json(prompt,8,name,True))
  else:call=json.loads(garnet.infer_json(prompt,image,8,name))
  row=dict(kind=kind,call=call);result['records'].append(row);save();assert call['status']=='ok';state=json.loads(garnet.native_text_worker_state_json(name));row['state']=state;save()
  assert state['states']==ref['states']and all(call[k]==ref['call'][k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])
  child=call['timings']['native_gateway']['child'];assert child['birth_id']==initial['birth_id']and child['infer_calls']==i+1==call['native_worker_call_index'];assert all(v==0 for v in call['timings']['native_gateway']['parent_primary_encoded_delta'].values())
  assert call['timings']['native_gateway']['route']=='native_lrpc_raw_text';assert len(state['states'])==28
 assert garnet.shutdown()and json.loads(garnet.native_text_worker_info_json(''))['children']==[]
 result.update(all_six_Galaxy_absent_image_kinds_exact_native_28layer_state_tokens_profile_no_retry=True,scope='FAULT-only C++ constructor hook; absent in measured libraries. Native correctness only, no WorldSense speed claim.');save();print('Native actual28layerstates/tokens/profile for default-C++invalid/None/number/dict/bytes/string exact; no retry PASS',flush=True)
finally:garnet.shutdown()
