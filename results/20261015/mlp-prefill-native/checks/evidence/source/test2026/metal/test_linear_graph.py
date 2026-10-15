"""Checkpoint-backed complete linear attention/decoder graphs, independent references."""
import json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi5b'))
from checkpoint_fixture import write_checkpoint
from reference_linear import step_linear
from reference_decoder import linear_block
rng=np.random.default_rng(353)
names=dict(qkv='linear_attn.in_proj_qkv.weight',z='linear_attn.in_proj_z.weight',b_proj='linear_attn.in_proj_b.weight',
 a_proj='linear_attn.in_proj_a.weight',conv='linear_attn.conv1d.weight',a_log='linear_attn.A_log',
 dt_bias='linear_attn.dt_bias',norm='linear_attn.norm.weight',out_proj='linear_attn.out_proj.weight')
common=dict(input_norm='input_layernorm.weight',post_norm='post_attention_layernorm.weight',
 mlp_gate='mlp.gate_proj.weight',mlp_up='mlp.up_proj.weight',mlp_down='mlp.down_proj.weight')
for batch in ((int(sys.argv[4]),) if len(sys.argv)>4 else (1,2)):
 for kh,vh in ((2,2),(1,2)):
  for decoder in (False,True):
   kd,vd,d,kernel=7,5,11,4;channels=2*kh*kd+vh*vd
   shapes=dict(qkv=(channels,d),z=(vh*vd,d),b_proj=(vh,d),a_proj=(vh,d),conv=(channels,1,kernel),
       a_log=(vh,),dt_bias=(vh,),norm=(vd,),out_proj=(d,vh*vd))
   suffixes=names.copy()
   if decoder:
    shapes.update(input_norm=(d,),post_norm=(d,),mlp_gate=(17,d),mlp_up=(17,d),mlp_down=(d,17));suffixes.update(common)
   weights={name:rng.normal(0,.2,shape).astype(np.float32) for name,shape in shapes.items()};weights['norm']+=1
   config=dict(batch=batch,decoder=decoder,hidden_size=d,linear_num_key_heads=kh,linear_num_value_heads=vh,
       linear_key_head_dim=kd,linear_value_head_dim=vd,linear_conv_kernel_dim=kernel,rms_norm_eps=1e-6)
   with tempfile.TemporaryDirectory(prefix='garnet-metal-linear-') as folder:
    folder=Path(folder);checkpoint=folder/'weights.safetensors'
    weights=write_checkpoint(checkpoint,weights,suffixes,'model.language_model.layers.0',('a_log','dt_bias','norm'))
    state=rng.normal(0,.2,(batch,vh,kd,vd)).astype(np.float32);conv=rng.normal(0,.2,(batch,channels,kernel)).astype(np.float32)
    steps=[]
    for i in range(7):
     x=rng.normal(size=(batch,d)).astype(np.float32);active=np.ones(batch,np.float32)
     if batch==2 and 2<=i<=4:active[1]=0
     inputs=[a.ravel().tolist() for a in (x,conv,state,active)]
     reference=linear_block if decoder else step_linear
     out,next_conv,next_state=reference(x,conv,state,active,weights,kh,vh,kd,vd)
     packed=np.concatenate([out,next_conv.reshape(batch,-1),next_state.reshape(batch,-1)],axis=-1)
     steps.append(dict(inputs=inputs,outputs=dict(packed=packed.ravel().tolist(),out=out.ravel().tolist(),
         conv_state=next_conv.ravel().tolist(),state=next_state.ravel().tolist())))
     conv,state=next_conv.astype(np.float32),next_state.astype(np.float32)
    for mode in ('packed','out','conv_state','state'):
     fixture=folder/'fixture.json';fixture.write_text(json.dumps(dict(config=dict(config,mode=mode),checkpoint=str(checkpoint),
        steps=[dict(inputs=s['inputs'],expected=s['outputs'][mode]) for s in steps])))
     subprocess.run([sys.argv[1],str(Path(__file__).with_name('run_linear_graph.py')),str(fixture),sys.argv[2],sys.argv[3]],check=True)
