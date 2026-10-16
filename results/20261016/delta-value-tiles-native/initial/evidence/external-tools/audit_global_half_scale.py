"""CPU-only feasibility: one lossless half scale for each actual body matrix."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();results=[]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
 assets=root/'logs/mps-gemv-probe-20261013/jobs'/(model+'-assets');manifest=json.loads((assets/'manifest.json').read_text())
 for c in manifest['cases']:
  if c['tensor'].startswith('synthetic') or c['columns']*c['inner']*2>16*1024*1024:continue
  raw=(assets/(c['name']+'.bin')).read_bytes();assert hashlib.sha256(raw).hexdigest()==c['sha256']
  bits=np.frombuffer(raw,dtype='<u2');mag=bits&32767;largest=int(mag.max());scale=142-(largest>>7) if largest else 0
  exponents=(mag>>7).astype(np.int32);e=exponents-127+scale;nonzero=mag!=0
  valid=(~nonzero)|((exponents>0)&(exponents<255)&(e>=-24)&(e<=15))
  sub=nonzero&(e<-14);discard=-(e[sub]+17);significands=128+(mag[sub]&127).astype(np.int64)
  fine=discard<=0
  for n in range(1,8):fine|=(discard==n)&((significands&((1<<n)-1))==0)
  valid[sub]&=fine;eligible=bool(valid.all()) and -64<=scale<=64
  item=dict(model=model,tensor=c['tensor'],columns=c['columns'],inner=c['inner'],bytes=len(raw),source_sha256=c['sha256'],global_scale=scale,eligible=eligible,unrepresentable_values=int((~valid).sum()))
  if eligible:
   f32=(bits.astype(np.uint32)<<16).view(np.float32);half=np.ldexp(f32,scale).astype(np.float16)
   restored=np.ldexp(half.astype(np.float32),-scale)
   assert np.array_equal(restored.view(np.uint32),f32.view(np.uint32))
   item.update(restored_bits_exact=True,encoded_half_sha256=hashlib.sha256(half.astype('<f2').tobytes()).hexdigest())
  results.append(item)
output=dict(scope=__doc__,numpy=np.__version__,matrices=results,eligible_count=sum(r['eligible'] for r in results),gpu_execution_performed=False)
target.write_bytes((json.dumps(output,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in output.items() if k!='matrices'}))
