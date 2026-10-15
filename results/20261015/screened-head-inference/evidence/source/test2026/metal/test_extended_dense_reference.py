"""Check real dense Qwen weights, paused rows and partial chunk tails."""
import sys,json,subprocess,tempfile
from pathlib import Path
import numpy as np
from reference_dense import weights,block
runtime,build,source,model=sys.argv[1:5];model=Path(model);c=json.loads((model/'config.json').read_text());c['head_dim']=c.get('head_dim',c['hidden_size']//c['num_attention_heads'])
rng=np.random.default_rng(765);capacity=256
for layer in (0,c['num_hidden_layers']-1):
 w=weights(model/'model.safetensors',layer,c)
 for batch in (1,2):
  for chunk in (32,64):
   key=np.zeros((batch,c['num_key_value_heads'],capacity,c['head_dim']),np.float32);value=np.zeros_like(key);base=np.zeros(batch,np.float32);steps=[]
   with tempfile.TemporaryDirectory(prefix='dense-metal-reference-') as temporary:
    folder=Path(temporary)
    for iteration in range(3):
     hidden=rng.normal(size=(batch,chunk,c['hidden_size'])).astype(np.float32);initial_key=key.copy();initial_value=value.copy();positions=np.zeros((batch,chunk,3),np.float32);slots=np.zeros((batch,chunk),np.float32);active=np.ones((batch,chunk),np.float32)
     if iteration==1:active[:,-1]=0
     if batch==2 and iteration==0:active[1,0]=0
     output=[]
     for token in range(chunk):
      if iteration==2 and token in (0,2):base=np.maximum(base-2,0)
      slots[:,token]=base;positions[:,token]=base[:,None]
      result,key,value=block(hidden[:,token],key,value,positions[:,token],slots[:,token],active[:,token],w,c);output.append(result);base+=active[:,token]
     inputs=[hidden,initial_key,initial_value,positions,slots,active];paths=[]
     for i,data in enumerate(inputs):
      path=folder/(str(iteration)+'-'+str(i)+'.f32');data.astype('<f4').tofile(path);paths.append(str(path))
     expected=np.concatenate([np.stack(output,axis=1).reshape(batch,-1),key.reshape(batch,-1),value.reshape(batch,-1)],axis=-1)
     path=folder/(str(iteration)+'-expected.f32');expected.astype('<f4').tofile(path);steps.append(dict(inputs=paths,expected=str(path),count=expected.size))
    invalid=[]
    for field,bad in [(5,2),(4,capacity),(3,float('nan'))]:
     data=np.fromfile(steps[0]['inputs'][field],dtype='<f4');data[0]=bad;path=folder/('invalid-'+str(field)+'.f32');data.tofile(path);files=list(steps[0]['inputs']);files[field]=str(path);invalid.append(files)
    fixture=folder/'fixture.json';fixture.write_text(json.dumps(dict(config=c,batch=batch,chunk=chunk,capacity=capacity,shapes=[list(a.shape) for a in inputs],layer=layer,checkpoint=str(model/'model.safetensors'),steps=steps,invalid_inputs=invalid)))
    subprocess.run([runtime,str(Path(__file__).with_name('run_dense_reference.py')),str(fixture),build,source]+sys.argv[5:],check=True)
