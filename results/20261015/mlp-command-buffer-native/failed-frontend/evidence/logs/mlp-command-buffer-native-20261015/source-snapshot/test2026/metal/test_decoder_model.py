"""Native I/O plus actual GPU decoder/head, checked through state feedback."""
import json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi5b'))
from checkpoint_fixture import write_checkpoint
from reference_decoder import linear_block,rms_norm
rng=np.random.default_rng(356)
suffixes={name:'layers.0.linear_attn.'+suffix for name,suffix in dict(qkv='in_proj_qkv.weight',z='in_proj_z.weight',
 b_proj='in_proj_b.weight',a_proj='in_proj_a.weight',conv='conv1d.weight',a_log='A_log',dt_bias='dt_bias',norm='norm.weight',out_proj='out_proj.weight').items()}
suffixes.update({name:'layers.0.'+suffix for name,suffix in dict(input_norm='input_layernorm.weight',post_norm='post_attention_layernorm.weight',
 mlp_gate='mlp.gate_proj.weight',mlp_up='mlp.up_proj.weight',mlp_down='mlp.down_proj.weight').items()})
suffixes.update(embedding='embed_tokens.weight',final_norm='norm.weight')
for batch in (1,2):
 kh,vh,kd,vd,d,kernel,vocab=1,2,7,5,11,4,19;channels=2*kh*kd+vh*vd
 shapes=dict(qkv=(channels,d),z=(vh*vd,d),b_proj=(vh,d),a_proj=(vh,d),conv=(channels,1,kernel),a_log=(vh,),
  dt_bias=(vh,),norm=(vd,),out_proj=(d,vh*vd),input_norm=(d,),post_norm=(d,),mlp_gate=(17,d),mlp_up=(17,d),
  mlp_down=(d,17),embedding=(vocab,d),final_norm=(d,))
 weights={name:rng.normal(0,.2,shape).astype(np.float32) for name,shape in shapes.items()};weights['norm']+=1
 config=dict(batch=batch,hidden_size=d,layer_types=['linear_attention'],rms_norm_eps=1e-6,linear_num_key_heads=kh,
  linear_num_value_heads=vh,linear_key_head_dim=kd,linear_value_head_dim=vd,linear_conv_kernel_dim=kernel)
 with tempfile.TemporaryDirectory(prefix='garnet-metal-decoder-') as folder:
  folder=Path(folder);checkpoint=folder/'weights.safetensors'
  weights=write_checkpoint(checkpoint,weights,suffixes,'model.language_model',('a_log','dt_bias','norm'))
  conv=np.zeros((batch,channels,kernel),np.float32);state=np.zeros((batch,vh,kd,vd),np.float32);steps=[]
  tokens=np.arange(batch,dtype=np.int64)+3
  for step in range(8):
   active=np.ones(batch,np.float32)
   if batch==2 and 2<=step<=4:active[1]=0
   embedding=weights['embedding'][tokens]
   hidden,conv,state=linear_block(embedding,conv,state,active,weights,kh,vh,kd,vd)
   logits=rms_norm(hidden,weights['final_norm'])@weights['embedding'].T
   predictions=np.argmax(logits,axis=-1)
   steps.append(dict(tokens=tokens.tolist(),active=active.tolist(),embedding=embedding.ravel().tolist(),prediction=predictions.tolist(),
       conv=conv.ravel().tolist(),state=state.ravel().tolist()))
   tokens=np.where(active!=0,predictions,tokens)
  fixture=folder/'fixture.json';fixture.write_text(json.dumps(dict(config=config,checkpoint=str(checkpoint),steps=steps)))
  subprocess.run([sys.argv[1],str(Path(__file__).with_name('run_decoder_model.py')),str(fixture),sys.argv[2],sys.argv[3]],check=True)
