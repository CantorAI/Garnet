"""Full-size decoder layers from the pinned checkpoint; not model generation."""
import json,mmap,resource,struct,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi5b'))
from reference_decoder import linear_block,attention_block
folder=Path(sys.argv[4]);config=json.loads((folder/'config.json').read_text())['text_config']
checkpoint=folder/'model.safetensors';rng=np.random.default_rng(355)
common=dict(input_norm='input_layernorm.weight',post_norm='post_attention_layernorm.weight',
 mlp_gate='mlp.gate_proj.weight',mlp_up='mlp.up_proj.weight',mlp_down='mlp.down_proj.weight')
def read_weights(names,layer):
 weights={}
 with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
  length=struct.unpack_from('<Q',mapping)[0];header=json.loads(mapping[8:8+length]);start=8+length
  for name,suffix in names.items():
   info=header['model.language_model.layers.'+str(layer)+'.'+suffix];a,b=info['data_offsets'];count=np.prod(info['shape'])
   if info['dtype']=='BF16':data=(np.frombuffer(mapping,dtype='<u2',count=int(count),offset=start+a).astype(np.uint32)<<16).view(np.float32)
   elif info['dtype']=='F32':data=np.frombuffer(mapping,dtype='<f4',count=int(count),offset=start+a).copy()
   else:raise ValueError('Unexpected pretrained weight dtype')
   weights[name]=data.reshape(info['shape'])
 return weights
for kind in ('linear','attention'):
 names=common.copy();d=config['hidden_size'];capacity=16;layer=0 if kind=='linear' else 3
 if kind=='linear':
  names.update({name:'linear_attn.'+suffix for name,suffix in dict(qkv='in_proj_qkv.weight',z='in_proj_z.weight',
   b_proj='in_proj_b.weight',a_proj='in_proj_a.weight',conv='conv1d.weight',a_log='A_log',dt_bias='dt_bias',
   norm='norm.weight',out_proj='out_proj.weight').items()})
  kh=config['linear_num_key_heads'];vh=config['linear_num_value_heads'];kd=config['linear_key_head_dim'];vd=config['linear_value_head_dim']
  channels=2*kh*kd+vh*vd
 else:
  names.update({name:'self_attn.'+suffix for name,suffix in dict(q='q_proj.weight',k='k_proj.weight',v='v_proj.weight',
   out='o_proj.weight',q_norm='q_norm.weight',k_norm='k_norm.weight').items()})
  heads=config['num_attention_heads'];kv=config['num_key_value_heads'];dim=config['head_dim'];rope=config['rope_parameters']
  rotary=int(dim*rope['partial_rotary_factor'])
 weights=read_weights(names,layer)
 for batch in (1,2):
  if kind=='linear':
   first=np.zeros((batch,channels,config['linear_conv_kernel_dim']),np.float32);second=np.zeros((batch,vh,kd,vd),np.float32)
  else:first=np.zeros((batch,kv,capacity,dim),np.float32);second=np.zeros_like(first)
  slots=np.zeros(batch,np.float32);steps=[]
  for step in range(4):
   hidden=rng.normal(size=(batch,d)).astype(np.float32);active=np.ones(batch,np.float32)
   if batch==2 and step==2:active[1]=0
   if kind=='linear':
    inputs=[a.ravel().tolist() for a in (hidden,first,second,active)]
    out,next_first,next_second=linear_block(hidden,first,second,active,weights,kh,vh,kd,vd)
   else:
    positions=np.stack([slots*2+1,slots*3+4,slots*5+7],axis=-1).astype(np.float32)
    inputs=[a.ravel().tolist() for a in (hidden,first,second,positions,slots,active)]
    out,next_first,next_second=attention_block(hidden,first,second,positions,slots,active,weights,heads,kv,dim,rotary,rope['mrope_section'],rope['rope_theta'])
   packed=np.concatenate([out,next_first.reshape(batch,-1),next_second.reshape(batch,-1)],axis=-1)
   steps.append(dict(inputs=inputs,expected=packed.ravel().tolist()))
   first,second=next_first.astype(np.float32),next_second.astype(np.float32);slots+=active
  with tempfile.TemporaryDirectory(prefix='garnet-metal-pretrained-') as tmp:
   fixture=Path(tmp)/'fixture.json';fixture.write_text(json.dumps(dict(config=dict(config,batch=batch,decoder=True,mode='packed',capacity=capacity),
       checkpoint=str(checkpoint),steps=steps)))
   subprocess.run([sys.argv[1],str(Path(__file__).with_name('run_'+kind+'_graph.py')),str(fixture),sys.argv[2],sys.argv[3]],check=True)
  rss=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
  print('Full-size pretrained',kind,'decoder batch',batch,'PASS child peak RSS MiB',rss,flush=True)
