import sys,json,pathlib,hashlib,struct
build,source,label=sys.argv[1:4];sys.path.insert(0,build)
import garnet
sys.path.insert(0,source)
from runtime.metal_serving import Model
m=Model('/Users/shawn/CantorAI/out/models/Qwen3.5-4B',source+'/xModel/qwen35/vl_4b','/Users/shawn/Documents/Codex/2026-10-15/i/work/4b-shape/state-cache-'+label,{'maxInputTokens':384,'maxOutputTokens':128,'prefillChunk':16,'linearChunk':True,'imagePixels':65536},'Qwen3.5-4B')
r=m.infer('Describe the visible subject in one sentence.','/Users/shawn/Documents/Codex/2026-10-15/i/work/skill-expansion/car.jpg',16)
def flat(v):
 for x in v:
  if isinstance(x,list):yield from flat(x)
  else:yield x
states=[]
for layer in range(32):
 parts=[]
 for t in m.decoder.session.snapshot(layer):
  h=hashlib.sha256();count=0
  for x in flat(t.tolist()):h.update(struct.pack('<f',x));count+=1
  parts.append({'values':count,'sha256':h.hexdigest()})
 states.append(parts)
p=pathlib.Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/4b-shape');(p/(label+'-states.json')).write_text(json.dumps({'token_ids':r['token_ids'],'states':states},indent=2));print('STATE_SNAPSHOT_COMPLETE',label,flush=True)
