"""Persist sparse active chunk fixtures for each frozen checkpoint and B1/B2."""
import json,subprocess,sys
from pathlib import Path
import numpy as np
runtime,build,source,model,output=sys.argv[1:6];model=Path(model);output=Path(output);output.mkdir()
raw=json.loads((model/'config.json').read_text());c=raw.get('text_config',raw);c['head_dim']=c.get('head_dim',c['hidden_size']//c['num_attention_heads']);rng=np.random.default_rng(151026)
for batch in (1,2):
 cache=[batch,c['num_key_value_heads'],8,c['head_dim']];shapes=[[batch,2,c['hidden_size']],cache,cache,[batch,2,3],[batch,2],[batch,2]]
 flags=np.ones((batch,2),np.float32);slots=np.tile([0,3],(batch,1)).astype(np.float32)
 if batch==2:flags[-1,-1]=0;slots[-1,-1]=7
 values=[rng.normal(0,.25,shapes[0]).astype(np.float32),np.zeros(cache,np.float32),np.zeros(cache,np.float32),np.tile([[0,0,0],[3,3,3]],(batch,1,1)).astype(np.float32),slots,flags];paths=[]
 for index,v in enumerate(values):
  path=output/(str(batch)+'-'+str(index)+'.f32');v.astype('<f4').tofile(path);paths.append(str(path))
 fixture=dict(config=c,vl='text_config' in raw,batch=batch,shapes=shapes,checkpoint=str(model/'model.safetensors'),inputs=paths);path=output/('fixture-b'+str(batch)+'.json');path.write_bytes((json.dumps(fixture,indent=2)+'\n').encode())
 subprocess.run([runtime,str(Path(__file__).with_name('run_prefix_reset_prefill_reference.py')),str(path),build,source],check=True)
