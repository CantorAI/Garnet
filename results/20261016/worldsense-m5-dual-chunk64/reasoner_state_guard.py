"""Native correctness-only policy/state guard. No speed results are emitted."""
import sys,pathlib,json,hashlib,struct
build='/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-dual-chunk64';source='/Users/shawn/CantorAI/out/research/worldsense-dual-chunk64/source';sys.path.insert(0,build);sys.path.insert(0,source)
from runtime.metal_serving import Model
m=Model('/Users/shawn/CantorAI/out/models/Qwen3-1.7B',source+'/xModel/qwen3/text_1_7b','/Users/shawn/CantorAI/out/cache/Qwen3-1.7B',{'maxInputTokens':384,'maxOutputTokens':128,'prefillChunk':16,'linearChunk':True,'imagePixels':65536},'Qwen3-1.7B')
records=[]
for enabled in [False,True]:
 m.decoder.chunk=64 if enabled else 16
 m.prefill_chunk=m.decoder.chunk
 call=m.infer('Return JSON describing whether the current evidence shows a passenger vehicle. Evidence: a red car travels past the stationary viewpoint. ' * 8,'',16)
 states=[]
 for layer in range(28):
  parts=[]
  for t in m.decoder.session.snapshot(layer):
   values=t.tolist();assert not values or not isinstance(values[0],list);h=hashlib.sha256()
   for i in range(0,len(values),4096):
    b=values[i:i+4096];h.update(struct.pack('<'+str(len(b))+'f',*b))
   parts.append({'values':len(values),'sha256':h.hexdigest()})
  states.append(parts)
 count=max(call['timings']['prefill_chunk_plan']);assert count==(64 if enabled else 16)
 records.append({'policy':'candidate'if enabled else'control','generated_token_ids':call['generated_token_ids'],'states':states,'actual_max_prefill_chunk':count})
assert records[0]['generated_token_ids']==records[1]['generated_token_ids'] and records[0]['states']==records[1]['states']
out=pathlib.Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-dual-chunk64/reasoner-state-guard.json');out.write_text(json.dumps({'all_28_state_layers_and_tokens_bitwise_equal':True,'records':records},indent=2));print('1.7B paired policy, actual chunk plan, unchanged decode path and 28-layer state bits PASS',flush=True)
