from pathlib import Path
import json,struct,hashlib
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');weights=[];headers={}
for model in ['Qwen3.5-4B','Qwen3-1.7B']:
 p=r/'out/models'/model;index=p/'model.safetensors.index.json';metadata=json.loads(index.read_text());headers[model]={}
 for shard in sorted(set(metadata['weight_map'].values())):
  with(p/shard).open('rb')as f:n=struct.unpack('<Q',f.read(8))[0];raw=f.read(n)
  headers[model][shard]=hashlib.sha256(raw).hexdigest()
  for name,info in sorted(json.loads(raw).items()):
   if name=='__metadata__'or info['dtype']!='BF16':continue
   shape=info['shape'];eligible=name.startswith('model.visual.')or len(shape)==1 or(len(shape)==3 and shape[1]==1)
   if not eligible:continue
   assert len(shape)in [1,2,3,5]and all(x>0 for x in shape)
   rows=1 if len(shape)==1 else shape[0];columns=1
   for x in (shape if len(shape)==1 else shape[1:]):columns*=x
   assert rows*columns*2==info['data_offsets'][1]-info['data_offsets'][0]
   weights.append(dict(model=model,checkpoint=str(index),tensor=name,shape=shape,rows=rows,columns=columns))
d=dict(weights=sorted(weights,key=lambda x:(x['model'],x['tensor'])),headers_sha256=headers,scope='All learned BF16 model.visual.* plus1Dnorm/bias and3Ddepthwise rawWeight candidates. No language2Dprojection/embeddingIO change claimed. Superset of actual graph accesses; completevision/state proof separately required.')
(w/'direct-upload-selection.json').write_text(json.dumps(d,indent=2)+'\n');print('Selected direct corpus weights',len(weights),'words',sum(x['rows']*x['columns']for x in weights))
