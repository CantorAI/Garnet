import json,mmap,struct,subprocess,sys,tempfile,shutil
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi5b'))
from reference_decoder import linear_block,attention_block
from qwen35_weight_reference import read_checkpoint_weights
folder=Path(sys.argv[4]);config=json.loads((folder/'config.json').read_text())['text_config']
checkpoint=folder/'model.safetensors';rng=np.random.default_rng(456)
if not checkpoint.exists():checkpoint=folder
common=dict(input_norm='input_layernorm.weight',post_norm='post_attention_layernorm.weight',
 mlp_gate='mlp.gate_proj.weight',mlp_up='mlp.up_proj.weight',mlp_down='mlp.down_proj.weight')
def read_weights(names,layer):
 return read_checkpoint_weights(folder,{name:'model.language_model.layers.'+str(layer)+'.'+suffix for name,suffix in names.items()})
for kind in ('linear','attention'):
 for layer in ((0,config['num_hidden_layers']-2) if kind=='linear' else (3,config['num_hidden_layers']-1)):
  names=common.copy();d=config['hidden_size'];capacity=256
  if kind=='linear':
   names.update({name:'linear_attn.'+suffix for name,suffix in dict(qkv='in_proj_qkv.weight',z='in_proj_z.weight',b_proj='in_proj_b.weight',a_proj='in_proj_a.weight',conv='conv1d.weight',a_log='A_log',dt_bias='dt_bias',norm='norm.weight',out_proj='out_proj.weight').items()})
   kh=config['linear_num_key_heads'];vh=config['linear_num_value_heads'];kd=config['linear_key_head_dim'];vd=config['linear_value_head_dim'];channels=2*kh*kd+vh*vd
  else:
   names.update({name:'self_attn.'+suffix for name,suffix in dict(q='q_proj.weight',k='k_proj.weight',v='v_proj.weight',out='o_proj.weight',q_norm='q_norm.weight',k_norm='k_norm.weight').items()})
   heads=config['num_attention_heads'];kv=config['num_key_value_heads'];dim=config['head_dim'];rope=config['rope_parameters'];rotary=int(dim*rope['partial_rotary_factor'])
  weights=read_weights(names,layer)
  for batch in (1,2):
   for chunk in ((40,80) if batch==1 else (40,)):
    if kind=='linear':first=np.zeros((batch,channels,config['linear_conv_kernel_dim']),np.float32);second=np.zeros((batch,vh,kd,vd),np.float32)
    else:first=np.zeros((batch,kv,capacity,dim),np.float32);second=np.zeros_like(first)
    base=np.zeros(batch,np.float32);steps=[]
    with tempfile.TemporaryDirectory(prefix='metal-chunk-') as temp:
     temp=Path(temp)
     for iteration in range(3):
      hidden=rng.normal(size=(batch,chunk,d)).astype(np.float32);positions=np.zeros((batch,chunk,3),np.float32);slots=np.zeros((batch,chunk),np.float32);active=np.ones((batch,chunk),np.float32)
      if iteration==1:active[:,-1]=0
      if batch==2 and iteration==0:active[1,0]=0
      initial_first=first.copy();initial_second=second.copy();outputs=[]
      for token in range(chunk):
       if iteration==2 and token in (0,2):base=np.maximum(base-2,0)
       slots[:,token]=base;positions[:,token,:]=np.stack([base*2+1,base*3+4,base*5+7],axis=-1)
       if kind=='linear':out,first,second=linear_block(hidden[:,token],first,second,active[:,token],weights,kh,vh,kd,vd)
       else:out,first,second=attention_block(hidden[:,token],first,second,positions[:,token],slots[:,token],active[:,token],weights,heads,kv,dim,rotary,rope['mrope_section'],rope['rope_theta'])
       first=first.astype(np.float32);second=second.astype(np.float32);outputs.append(out);base+=active[:,token]
      inputs=[hidden,initial_first,initial_second,positions,slots,active];paths=[]
      for index,data in enumerate(inputs):
       path=temp/(str(iteration)+'-'+str(index)+'.f32');data.astype('<f4').tofile(path);paths.append(str(path))
      expected=np.concatenate([np.stack(outputs,axis=1).reshape(batch,-1),first.reshape(batch,-1),second.reshape(batch,-1)],axis=-1)
      path=temp/(str(iteration)+'-expected.f32');expected.astype('<f4').tofile(path);steps.append(dict(inputs=paths,expected=str(path),count=expected.size))
     invalid=[]
     for field,value in [(5,2.0)]+([(4,float(capacity)),(3,float('nan'))] if kind=='attention' else []):
      data=np.fromfile(steps[0]['inputs'][field],dtype='<f4');data[0]=value
      path=temp/('invalid-'+str(field)+'.f32');data.tofile(path);files=list(steps[0]['inputs']);files[field]=str(path);invalid.append(files)
     fixture=temp/'fixture.json';fixture.write_text(json.dumps(dict(config=config,batch=batch,chunk=chunk,capacity=capacity,shapes=[list(a.shape) for a in inputs],kind='linear_attention' if kind=='linear' else 'full_attention',layer=layer,checkpoint=str(checkpoint),steps=steps,invalid_inputs=invalid)))
     command=[sys.argv[1],str(Path(__file__).with_name('run_chunk_graph.py')),str(fixture),sys.argv[2],sys.argv[3],'bf16','linear-chunk']
     result=subprocess.run(command)
     if result.returncode:
      target=Path(sys.argv[5])/('layer-'+str(layer)+'-b'+str(batch)+'-c'+str(chunk))
      shutil.copytree(temp,target)
      data=json.loads((target/'fixture.json').read_text())
      for step in data['steps']:
       step['inputs']=[str(target/Path(p).name) for p in step['inputs']]
       step['expected']=str(target/Path(step['expected']).name)
      data['invalid_inputs']=[[str(target/Path(p).name) for p in files] for files in data['invalid_inputs']]
      (target/'fixture.json').write_text(json.dumps(data,indent=2))
      raise RuntimeError('Extended reference failed; fixture retained at '+str(target))
