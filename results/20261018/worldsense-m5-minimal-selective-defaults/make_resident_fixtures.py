"""Host-only deterministic numeric correctness fixtures; no image processing/timing."""
from pathlib import Path
import json,random,struct,math
w=Path(__file__).resolve().parent;model=Path('/Users/shawn/CantorAI/out/models/Qwen3-1.7B');c=json.loads((model/'config.json').read_text());c['head_dim']=c.get('head_dim',c['hidden_size']//c['num_attention_heads']);rng=random.Random(776)
for batch,capacity in [(1,8),(2,64)]:
 folder=w/f'fixture-{batch}-{capacity}';folder.mkdir(exist_ok=True);cache=[batch,c['num_key_value_heads'],capacity,c['head_dim']];shapes=[[batch,c['hidden_size']],cache,cache,[batch,3],[batch],[batch]];steps=[]
 def write(name,values):
  p=folder/name;p.write_bytes(struct.pack('<'+'f'*len(values),*values));return str(p)
 for i,slot in enumerate((0,3,capacity-1,1,3,0,2,capacity-1)):
  flags=[1.]*batch
  if i in (1,5):flags[-1]=0.
  if i==3:flags=[0.]*batch
  values=[[rng.gauss(0,.5)for _ in range(batch*c['hidden_size'])],[0.]*math.prod(cache),[0.]*math.prod(cache),[float(x)for _ in range(batch)for x in (slot*2+1,slot*3+4,slot*5+7)],[float(slot)]*batch,flags]
  steps.append([write(f'{i}-{j}.f32',v)for j,v in enumerate(values)])
 invalid=[]
 for field,bad in ((5,2.),(4,float(capacity)),(3,float('nan'))):
  paths=list(steps[0]);raw=Path(paths[field]).read_bytes();v=list(struct.unpack('<'+'f'*(len(raw)//4),raw));v[0]=bad;paths[field]=write(f'invalid-{field}.f32',v);invalid.append(paths)
 (folder/'fixture.json').write_text(json.dumps(dict(config=c,vl=False,batch=batch,capacity=capacity,layer=0,shapes=shapes,steps=steps,invalid=invalid,checkpoint=str(model/'model.safetensors.index.json'))))
