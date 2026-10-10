"""Pretrained layer-3 gate/mRoPE/KV validation, not full model inference."""
import json,resource,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from checkpoint_fixture import read_checkpoint
from reference_attention import step_attention
checkpoint=Path(sys.argv[3]).resolve()
suffixes=dict(q='q_proj.weight',k='k_proj.weight',v='v_proj.weight',out='o_proj.weight',q_norm='q_norm.weight',k_norm='k_norm.weight')
weights=read_checkpoint(checkpoint,suffixes,'model.language_model.layers.3.self_attn')
rng=np.random.default_rng(534)
heads,kv,dim,rotary,d,capacity=8,2,256,64,1024,8;sections=[11,11,10];theta=10000000
for batch in (1,2):
    config=dict(batch=batch,mode='packed',hidden_size=d,num_attention_heads=heads,num_key_value_heads=kv,head_dim=dim,capacity=capacity,
        layer_idx=3,rms_norm_eps=1e-6,rope_parameters=dict(partial_rotary_factor=.25,rope_theta=theta,mrope_section=sections))
    key=np.zeros((batch,kv,capacity,dim),dtype=np.float32);value=np.zeros_like(key);slots=np.zeros(batch,dtype=np.float32)
    steps=[]
    for step in range(4):
        x=rng.normal(size=(batch,d)).astype(np.float32);active=np.ones(batch,dtype=np.float32)
        if batch==2 and step==2:active[1]=0
        positions=np.stack((slots*2+1,slots*3+4,slots*5+7),axis=-1).astype(np.float32)
        inputs=[a.ravel().tolist() for a in (x,key,value,positions,slots,active)]
        out,next_key,next_value=step_attention(x,key,value,positions,slots,active,weights,heads,kv,dim,rotary,sections,theta)
        packed=np.concatenate((out,next_key.reshape(batch,-1),next_value.reshape(batch,-1)),axis=1)
        steps.append(dict(inputs=inputs,expected=packed.ravel().tolist()))
        key,value=next_key,next_value;slots+=active
    with tempfile.TemporaryDirectory(prefix='garnet-pretrained-attention-') as folder:
        fixture=dict(config=config,checkpoint=str(checkpoint),steps=steps,rtol=5e-5,atol=5e-6)
        path=Path(folder)/'fixture.json';path.write_text(json.dumps(fixture))
        subprocess.run([sys.argv[1],sys.argv[2],str(Path(__file__).with_name('run_attention_graph.py')),str(path)],check=True)
        print('native probe cumulative max RSS KiB',resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,flush=True)
    print('pretrained full-size gated attention layer batch',batch,'PASS',flush=True)
