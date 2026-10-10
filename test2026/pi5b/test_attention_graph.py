import json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from checkpoint_fixture import write_checkpoint
from reference_attention import step_attention
rng=np.random.default_rng(533)
suffixes=dict(q='q_proj.weight',k='k_proj.weight',v='v_proj.weight',out='o_proj.weight',q_norm='q_norm.weight',k_norm='k_norm.weight')
for batch in (1,2):
    for heads,kv in ((4,2),(2,2)):
        dim,rotary,d,capacity=12,8,11,8;sections=[2,1,1];theta=10000000
        shapes=dict(q=(2*heads*dim,d),k=(kv*dim,d),v=(kv*dim,d),out=(d,heads*dim),q_norm=(dim,),k_norm=(dim,))
        weights={name:rng.normal(0,.2,size=shape).astype(np.float32) for name,shape in shapes.items()}
        with tempfile.TemporaryDirectory(prefix='garnet-attention-') as folder:
            checkpoint=Path(folder)/'weights.safetensors'
            weights=write_checkpoint(checkpoint,weights,suffixes,'model.language_model.layers.3.self_attn')
            config=dict(batch=batch,hidden_size=d,num_attention_heads=heads,num_key_value_heads=kv,head_dim=dim,capacity=capacity,layer_idx=3,
                rms_norm_eps=1e-6,rope_parameters=dict(partial_rotary_factor=rotary/dim,rope_theta=theta,mrope_section=sections))
            key=rng.normal(0,.3,size=(batch,kv,capacity,dim)).astype(np.float32);value=rng.normal(0,.3,size=key.shape).astype(np.float32)
            slots=np.zeros(batch,dtype=np.float32);steps=[]
            for step in range(6):
                x=rng.normal(size=(batch,d)).astype(np.float32);active=np.ones(batch,dtype=np.float32)
                if batch==2 and 2<=step<=3:active[1]=0
                positions=np.stack((slots*2+1,slots*3+4,slots*5+7),axis=-1).astype(np.float32)
                inputs=[a.ravel().tolist() for a in (x,key,value,positions,slots,active)]
                out,next_key,next_value=step_attention(x,key,value,positions,slots,active,weights,heads,kv,dim,rotary,sections,theta)
                packed=np.concatenate((out,next_key.reshape(batch,-1),next_value.reshape(batch,-1)),axis=1)
                steps.append(dict(inputs=inputs,outputs=dict(out=out.ravel().tolist(),key_cache=next_key.ravel().tolist(),
                    value_cache=next_value.ravel().tolist(),packed=packed.ravel().tolist())))
                key,value=next_key,next_value;slots+=active
            for mode in ('packed','out','key_cache','value_cache'):
                fixture=dict(config=dict(config,mode=mode),checkpoint=str(checkpoint),steps=[dict(inputs=s['inputs'],expected=s['outputs'][mode]) for s in steps])
                path=Path(folder)/'fixture.json';path.write_text(json.dumps(fixture))
                subprocess.run([sys.argv[1],sys.argv[2],str(Path(__file__).with_name('run_attention_graph.py')),str(path)],check=True)
        print('captured gated full attention, partial mRoPE, KV isolation/extraction batch',batch,'heads',heads,kv,'PASS',flush=True)
