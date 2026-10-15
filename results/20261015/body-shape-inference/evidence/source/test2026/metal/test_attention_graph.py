"""Actual full-attention xModel capture compared with independent NumPy."""
import json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi5b'))
from checkpoint_fixture import write_checkpoint
from reference_attention import step_attention
from reference_decoder import attention_block
rng=np.random.default_rng(354)
names=dict(q='self_attn.q_proj.weight',k='self_attn.k_proj.weight',v='self_attn.v_proj.weight',
 out='self_attn.o_proj.weight',q_norm='self_attn.q_norm.weight',k_norm='self_attn.k_norm.weight')
common=dict(input_norm='input_layernorm.weight',post_norm='post_attention_layernorm.weight',
 mlp_gate='mlp.gate_proj.weight',mlp_up='mlp.up_proj.weight',mlp_down='mlp.down_proj.weight')
for batch in ((int(sys.argv[4]),) if len(sys.argv)>4 else (1,2)):
 for rotary,sections in ((8,[2,1,1]),(12,[2,2,2])):
  for decoder in (False,True):
   d,heads,kv,dim,capacity,theta=11,4,2,12,8,10000000
   shapes=dict(q=(2*heads*dim,d),k=(kv*dim,d),v=(kv*dim,d),out=(d,heads*dim),q_norm=(dim,),k_norm=(dim,))
   suffixes=names.copy()
   if decoder:shapes.update(input_norm=(d,),post_norm=(d,),mlp_gate=(17,d),mlp_up=(17,d),mlp_down=(d,17));suffixes.update(common)
   weights={name:rng.normal(0,.2,shape).astype(np.float32) for name,shape in shapes.items()}
   config=dict(batch=batch,decoder=decoder,hidden_size=d,num_attention_heads=heads,num_key_value_heads=kv,head_dim=dim,
       capacity=capacity,rms_norm_eps=1e-6,rope_parameters=dict(partial_rotary_factor=rotary/dim,rope_theta=theta,mrope_section=sections))
   with tempfile.TemporaryDirectory(prefix='garnet-metal-attention-') as folder:
    folder=Path(folder);checkpoint=folder/'weights.safetensors'
    weights=write_checkpoint(checkpoint,weights,suffixes,'model.language_model.layers.3')
    key=rng.normal(0,.2,(batch,kv,capacity,dim)).astype(np.float32);value=rng.normal(0,.2,key.shape).astype(np.float32)
    slots=np.zeros(batch,np.float32);steps=[]
    for i in range(7):
     hidden=rng.normal(size=(batch,d)).astype(np.float32);active=np.ones(batch,np.float32)
     if batch==2 and 2<=i<=4:active[1]=0
     positions=np.stack([slots*2+1,slots*3+4,slots*5+7],axis=-1).astype(np.float32)
     inputs=[a.ravel().tolist() for a in (hidden,key,value,positions,slots,active)]
     reference=attention_block if decoder else step_attention
     out,next_key,next_value=reference(hidden,key,value,positions,slots,active,weights,heads,kv,dim,rotary,sections,theta)
     packed=np.concatenate([out,next_key.reshape(batch,-1),next_value.reshape(batch,-1)],axis=-1)
     steps.append(dict(inputs=inputs,outputs=dict(packed=packed.ravel().tolist(),out=out.ravel().tolist(),
         key=next_key.ravel().tolist(),value=next_value.ravel().tolist())))
     key,value=next_key.astype(np.float32),next_value.astype(np.float32);slots+=active
    for mode in ('packed','out','key','value'):
     fixture=folder/'fixture.json';fixture.write_text(json.dumps(dict(config=dict(config,mode=mode),checkpoint=str(checkpoint),
        steps=[dict(inputs=s['inputs'],expected=s['outputs'][mode]) for s in steps])))
     subprocess.run([sys.argv[1],str(Path(__file__).with_name('run_attention_graph.py')),str(fixture),sys.argv[2],sys.argv[3]],check=True)
