"""Correctness-only guard, not performance evidence. Execute with XLang3."""
import sys,json,pathlib,hashlib,struct
build,source,image,output=sys.argv[1:5];sys.path.insert(0,build);sys.path.insert(0,source)
from runtime.metal_serving import Model
m=Model('/Users/shawn/CantorAI/out/models/Qwen3.5-4B',source+'/xModel/qwen35/vl_4b','/Users/shawn/CantorAI/out/cache/Qwen3.5-4B',{'maxInputTokens':384,'maxOutputTokens':128,'prefillChunk':16,'linearChunk':True,'imagePixels':65536},'Qwen3.5-4B')
r=m.infer('Describe the visible subject in one sentence.',image,16)
def flat(values):
 for value in values:
  if isinstance(value,list):yield from flat(value)
  else:yield value
states=[]
for layer in range(32):
 parts=[]
 for t in m.decoder.session.snapshot(layer):
  values=t.tolist()
  if values and isinstance(values[0],list):values=list(flat(values))
  h=hashlib.sha256()
  for i in range(0,len(values),4096):
   block=values[i:i+4096];h.update(struct.pack('<'+str(len(block))+'f',*block))
  parts.append({'values':len(values),'sha256':h.hexdigest()})
 states.append(parts)
pathlib.Path(output).write_text(json.dumps({'generated_token_ids':r['generated_token_ids'],'states':states,'timings':r['timings']},indent=2));print('4B 32-layer state snapshot saved',flush=True)
