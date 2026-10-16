"""Persist deterministic validation fixtures; no serving timing."""
import json,subprocess,sys
from pathlib import Path
import numpy as np
runtime,build,source,model,output=sys.argv[1:6];model=Path(model);output=Path(output);output.mkdir()
raw=json.loads((model/'config.json').read_text());vl='text_config' in raw;c=raw.get('text_config',raw);c['head_dim']=c.get('head_dim',c['hidden_size']//c['num_attention_heads']);rng=np.random.default_rng(141026)
for batch in (1,2):
 capacity=8;cache=[batch,c['num_key_value_heads'],capacity,c['head_dim']];shapes=[[batch,c['hidden_size']],cache,cache,[batch,3],[batch],[batch]];fixture=dict(config=c,vl=vl,batch=batch,capacity=capacity,shapes=shapes,checkpoint=str(model/'model.safetensors'))
 for name in ('partial','nonfinite'):
  hidden=rng.normal(0,.25,shapes[0]).astype(np.float32)
  if name=='nonfinite':hidden[:]=np.nan
  flags=np.ones(batch,np.float32);slots=np.full(batch,3,np.float32)
  if batch==2:flags[-1]=0;slots[-1]=7
  values=[hidden,np.zeros(cache,np.float32),np.zeros(cache,np.float32),np.tile([3,3,3],(batch,1)).astype(np.float32),slots,flags];paths=[]
  for index,v in enumerate(values):
   path=output/(str(batch)+'-'+name+'-'+str(index)+'.f32');v.astype('<f4').tofile(path);paths.append(str(path))
  fixture[name]=paths
 path=output/('fixture-b'+str(batch)+'.json');path.write_bytes((json.dumps(fixture,indent=2)+'\n').encode())
 subprocess.run([runtime,str(Path(__file__).with_name('run_prefix_reset_session_reference.py')),str(path),build,source],check=True)
