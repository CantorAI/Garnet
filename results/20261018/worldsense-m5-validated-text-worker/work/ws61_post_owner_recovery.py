"""Native GPU/state recovery after owned-parent death, not timing evidence."""
import sys,json,os,hashlib
from pathlib import Path
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=root/'work/worldsense-validated-text-worker';s=Path('/Users/shawn/CantorAI/out/research/worldsense-validated-text-worker/source');b=Path('/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-validated-text-worker-on');sys.path.insert(0,str(s));sys.path.insert(0,str(b));import garnet
sys.path.insert(0,str(w))
from validated_call_guard import validated_call
assert json.loads((w/'owner-death-result.json').read_text())['all_three_parent_death_cases_pass']
name='Qwen3-1.7B';profile=dict(maxInputTokens=192,maxOutputTokens=64,prefillChunk=16,textPrefixCache=True)
base='Decide whether the visual facts satisfy the required condition. Use only the observation as evidence. Missing or contradictory evidence means matched=false. Do not substitute a different subject or condition. A still image cannot prove a transition. Return only a compact JSON object with matched and reason. '*2;prompt=base+'Observation: a red car passes a stationary viewpoint. Condition: a passenger vehicle is visible.'
result={}
def save():(w/'post-owner-death-recovery.json').write_text(json.dumps(result,indent=2)+'\n')
try:
 result['serve']=json.loads(garnet.serve_model('/Users/shawn/CantorAI/out/models/'+name,str(s/'xModel/qwen/metal_dense'),'/Users/shawn/CantorAI/out/cache/'+name,json.dumps(profile),name));save();assert result['serve']['ready']
 result['call']=json.loads(garnet.infer_json(prompt,'',8,name));save();assert result['call']['status']=='ok';validated_call(result['call'],'on');result['state']=json.loads(garnet.native_text_worker_state_json(name));save()
 ref=next(r for r in json.loads((w.parent/'worldsense-isolated-primary/api-state-guard-off.json').read_text())['records']if r['model_id']==name)
 assert all(result['call'][k]==ref['call'][k]for k in ['generated_token_ids','text','input_prompt_sha256','serving_configuration'])and result['state']['states']==ref['states']
 assert garnet.shutdown()and json.loads(garnet.native_text_worker_info_json(''))['children']==[]
 result.update(exact_28layer_native_state_tokens_text_prompt_profile_after_active_owner_death_pass=True,scope='Post-fault functional/numerical recovery only, no throughput claim.');save();print('Native GPU post-owner-death exact28layerstate/tokens/profile recovery PASS',flush=True)
finally:garnet.shutdown()
