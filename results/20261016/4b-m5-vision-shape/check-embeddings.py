import sys,json,pathlib,hashlib,struct,time
build,source,out=sys.argv[1:4];sys.path.insert(0,build)
import garnet
sys.path.insert(0,source)
from xModel.qwen35.vl_0_8b.metal_vision import MetalImageEncoder
root=pathlib.Path('/Users/shawn/CantorAI/out/models/Qwen3.5-4B');config=json.loads((root/'config.json').read_text())['vision_config'];checkpoint=str(root/'model.safetensors.index.json');prior=None;records=[]
def bits(t):
 def flat(v):
  for x in v:
   if isinstance(x,list):yield from flat(x)
   else:yield x
 return b''.join(struct.pack('<f',v) for v in flat(t.tolist()))
for name in ['bunny','car','traffic','bunny']:
 base=pathlib.Path(out)/name;gh,gw=json.loads(base.with_suffix('.json').read_text())['grid']
 fresh=MetalImageEncoder(checkpoint,config,gh,gw);expected=bits(fresh.encode_file(str(base.with_suffix('.f32'))));fresh.close()
 start=time.monotonic();replacement=MetalImageEncoder(checkpoint,config,gh,gw,shared_encoder=prior);seconds=time.monotonic()-start
 if prior is not None:prior.close()
 actual=bits(replacement.encode_file(str(base.with_suffix('.f32'))));assert actual==expected,'Embedding bits differ'
 records.append({'image':name,'grid':[gh,gw],'f32_values':len(actual)//4,'embedding_sha256':hashlib.sha256(actual).hexdigest(),'bitwise_equal':True,'shared_build_seconds':seconds})
 prior=replacement
prior.close()
try:MetalImageEncoder(checkpoint,config,gh,gw,shared_encoder=prior)
except Exception:closed_rejected=True
else:raise AssertionError('Closed source was accepted')
pathlib.Path(out,'embedding-checks.json').write_text(json.dumps({'records':records,'closed_source_rejected':closed_rejected},indent=2));print('EMBEDDING_CHECKS_PASS',flush=True)
