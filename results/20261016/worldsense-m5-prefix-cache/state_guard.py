"""Native correctness-only policy/state guard. No speed results are emitted."""
import sys,pathlib,json,hashlib,struct,math
build='/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-prefix-cache';source='/Users/shawn/CantorAI/out/research/worldsense-prefix-cache/source';sys.path.insert(0,build);sys.path.insert(0,source)
from runtime.metal_serving import Model
m=Model('/Users/shawn/CantorAI/out/models/Qwen3.5-4B',source+'/xModel/qwen35/vl_4b','/Users/shawn/CantorAI/out/cache/Qwen3.5-4B',{'maxInputTokens':384,'maxOutputTokens':128,'prefillChunk':16,'linearChunk':True,'imagePixels':65536},'Qwen3.5-4B')
records=[]
for enabled in [False,True]:
 m.decoder.chunk=64 if enabled else 16
 m.prefill_chunk=m.decoder.chunk
 call=m.infer('Describe the visible subject in one sentence.','/Users/shawn/Documents/Codex/2026-10-15/i/work/skill-expansion/car.jpg',16)
 states=[]
 for layer in range(32):
  parts=[]
  for t in m.decoder.session.snapshot(layer):
   values=t.tolist();assert not values or not isinstance(values[0],list);assert all(math.isfinite(v)for v in values);h=hashlib.sha256()
   for i in range(0,len(values),4096):
    b=values[i:i+4096];h.update(struct.pack('<'+str(len(b))+'f',*b))
   parts.append({'values':len(values),'sha256':h.hexdigest()})
  states.append(parts)
 count=max(call['timings']['prefill_chunk_plan']);assert count==(64 if enabled else 16)
 records.append({'policy':'candidate'if enabled else'control','generated_token_ids':call['generated_token_ids'],'states':states,'actual_max_prefill_chunk':count})
assert records[0]['generated_token_ids']==records[1]['generated_token_ids'] and records[0]['states']==records[1]['states']
overflow_rejected=False
try:m.infer('word ' * 4096,'',16)
except ValueError as error:
 assert 'capacity' in str(error);overflow_rejected=True
assert overflow_rejected
recovered=m.infer('Describe the visible subject in one sentence.','/Users/shawn/Documents/Codex/2026-10-15/i/work/skill-expansion/car.jpg',16)
assert recovered['generated_token_ids']==records[1]['generated_token_ids']
rejected=False
try:m.decoder.session.capture_prefix(64)
except Exception:rejected=True
assert rejected and m.decoder.session.prefix_cache_bytes==0
out=pathlib.Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-prefix-cache/state-guard.json');out.write_text(json.dumps({'all_32_state_layers_and_tokens_bitwise_equal':True,'all_state_values_finite':True,'capacity_rejected_and_recovered':True,'records':records},indent=2));print('4B paired policy, actual chunk plan, unchanged decode path and 32-layer state bits PASS',flush=True)
