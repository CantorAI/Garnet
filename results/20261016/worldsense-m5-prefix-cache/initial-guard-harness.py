"""Native/XLang3 correctness only: no speed results may be used from this run."""
import sys,json,pathlib,hashlib,struct,math
build='/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-prefix-cache';source='/Users/shawn/CantorAI/out/research/worldsense-prefix-cache/source';sys.path.insert(0,build);sys.path.insert(0,source)
import garnet
from runtime.metal_serving import Model
w=pathlib.Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-prefix-cache')
m=Model('/Users/shawn/CantorAI/out/models/Qwen3-1.7B',source+'/xModel/qwen/metal_dense','/Users/shawn/CantorAI/out/cache/Qwen3-1.7B',{'maxInputTokens':192,'maxOutputTokens':64,'prefillChunk':16},'Qwen3-1.7B')
base='Decide whether the visual facts satisfy the required condition. Use only the observation as evidence. Missing or contradictory evidence means matched=false. Do not substitute a different subject or condition. A still image cannot prove a transition. Return only a compact JSON object with matched and reason. '
prompts=[base+'Observation: a red car passes a stationary viewpoint. Condition: a passenger vehicle is visible.',base+'Observation: people walk across a street. Condition: a passenger vehicle is visible.']
def states():
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
records=[]
for enabled,index in [(False,0),(False,1),(True,0),(True,1),(True,0)]:
 m.text_prefix_cache=enabled;call=m.infer(prompts[index],'',8);record={'policy':enabled,'prompt_index':index,'generated_token_ids':call['generated_token_ids'],'states':states(),'timings':call['timings'],'prompt_sha256':call['input_prompt_sha256']};records.append(record)
 if enabled:
  ref=records[index];assert record['generated_token_ids']==ref['generated_token_ids']and record['states']==ref['states'],'Cached prefix changed full model states or output'
 if enabled and len(records)>=4:
  assert call['timings']['prefix_cache_hit']and call['timings']['prefix_cache_restored_tokens']==64 and call['timings']['prefix_restore_native_bytes']>0
 assert call['timings']['input_tokens']==call['timings']['executed_prefill_tokens']+call['timings']['prefix_cache_restored_tokens']
(w/'prefix-model-guard.json').write_text(json.dumps({'all_28_state_layers_and_tokens_bitwise_equal':True,'all_values_finite':True,'changing_suffix_cold_and_warm_hits':True,'records':records},indent=2));print('1.7B complete28-layer state/token bits and finite values, changing suffix, cold capture/warm restores PASS',flush=True)
m.decoder.close()
