"""Real trained single-token attention graphs with sparse private-cache writes."""
import json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
runtime,build,source,model=sys.argv[1:5];model=Path(model);raw=json.loads((model/'config.json').read_text())
vl='text_config' in raw;c=raw['text_config'] if vl else raw
c['head_dim']=c.get('head_dim',c['hidden_size']//c['num_attention_heads']);rng=np.random.default_rng(776)
for layer in ((3,c['num_hidden_layers']-1) if vl else (0,c['num_hidden_layers']-1)):
    for batch in (1,2):
        for capacity in (8,256):
            cache=[batch,c['num_key_value_heads'],capacity,c['head_dim']]
            shapes=[[batch,c['hidden_size']],cache,cache,[batch,3],[batch],[batch]]
            with tempfile.TemporaryDirectory(prefix='resident-kv-reference-') as temporary:
                folder=Path(temporary);steps=[]
                for i,slot in enumerate((0,3,capacity-1,1,3,0,2,capacity-1)):
                    hidden=rng.normal(0,.5,size=shapes[0]).astype(np.float32);flags=np.ones(batch,np.float32)
                    if i in (1,5):flags[-1]=0
                    if i==3:flags[:]=0
                    slots=np.full(batch,slot,np.float32);positions=np.tile([slot*2+1,slot*3+4,slot*5+7],(batch,1)).astype(np.float32)
                    values=[hidden,np.zeros(cache,np.float32),np.zeros(cache,np.float32),positions,slots,flags];paths=[]
                    for j,v in enumerate(values):
                        p=folder/f'{i}-{j}.f32';v.astype('<f4').tofile(p);paths.append(str(p))
                    steps.append(paths)
                invalid=[]
                for field,bad in ((5,2),(4,capacity),(3,float('nan'))):
                    paths=list(steps[0]);v=np.fromfile(paths[field],dtype='<f4');v[0]=bad;p=folder/f'invalid-{field}.f32';v.tofile(p);paths[field]=str(p);invalid.append(paths)
                fixture=folder/'fixture.json';fixture.write_text(json.dumps(dict(config=c,vl=vl,batch=batch,capacity=capacity,layer=layer,shapes=shapes,steps=steps,invalid=invalid,checkpoint=str(model/'model.safetensors'))))
                subprocess.run([runtime,str(Path(__file__).with_name('run_resident_kv_reference.py')),str(fixture),build,source],check=True)
