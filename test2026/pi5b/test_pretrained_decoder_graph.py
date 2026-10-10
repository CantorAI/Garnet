"""Pretrained complete blocks via captured Garnet graphs or native CPU package."""
import json,resource,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from checkpoint_fixture import read_checkpoint
from reference_decoder import linear_block,attention_block
rng=np.random.default_rng(536)
folder=Path(sys.argv[3]).resolve()
common=dict(input_norm='input_layernorm.weight',post_norm='post_attention_layernorm.weight',
    mlp_gate='mlp.gate_proj.weight',mlp_up='mlp.up_proj.weight',mlp_down='mlp.down_proj.weight')
for kind,layer in (('linear',0),('full',3)):
    checkpoint=folder/('block'+str(layer)+'.safetensors');suffixes=common.copy()
    if kind=='linear':
        names=dict(qkv='in_proj_qkv.weight',z='in_proj_z.weight',b_proj='in_proj_b.weight',a_proj='in_proj_a.weight',
            conv='conv1d.weight',a_log='A_log',dt_bias='dt_bias',norm='norm.weight',out_proj='out_proj.weight')
        suffixes.update({name:'linear_attn.'+suffix for name,suffix in names.items()})
    else:
        names=dict(q='q_proj.weight',k='k_proj.weight',v='v_proj.weight',out='o_proj.weight',q_norm='q_norm.weight',k_norm='k_norm.weight')
        suffixes.update({name:'self_attn.'+suffix for name,suffix in names.items()})
    weights=read_checkpoint(checkpoint,suffixes,'model.language_model.layers.'+str(layer))
    for batch in (1,2):
        d,mid=1024,3584;kh=vh=16;kd=vd=128;ks=4;channels=6144
        heads,kv,dim,rotary,capacity=8,2,256,64,8;sections=[11,11,10];theta=10000000
        config=dict(kind=kind,batch=batch,layer_idx=layer,hidden_size=d,intermediate_size=mid,rms_norm_eps=1e-6,
            linear_num_key_heads=kh,linear_num_value_heads=vh,linear_key_head_dim=kd,linear_value_head_dim=vd,linear_conv_kernel_dim=ks,
            num_attention_heads=heads,num_key_value_heads=kv,head_dim=dim,capacity=capacity,
            rope_parameters=dict(partial_rotary_factor=.25,rope_theta=theta,mrope_section=sections))
        if kind=='linear':cache1=np.zeros((batch,channels,ks),dtype=np.float32);cache2=np.zeros((batch,vh,kd,vd),dtype=np.float32)
        else:cache1=np.zeros((batch,kv,capacity,dim),dtype=np.float32);cache2=np.zeros_like(cache1)
        slots=np.zeros(batch,dtype=np.float32);steps=[]
        for step in range(4):
            x=rng.normal(size=(batch,d)).astype(np.float32);active=np.ones(batch,dtype=np.float32)
            if batch==2 and step==2:active[1]=0
            if kind=='linear':
                inputs=[a.ravel().tolist() for a in (x,cache1,cache2,active)]
                out,next1,next2=linear_block(x,cache1,cache2,active,weights,kh,vh,kd,vd)
            else:
                positions=np.stack((slots*2+1,slots*3+4,slots*5+7),axis=-1).astype(np.float32)
                inputs=[a.ravel().tolist() for a in (x,cache1,cache2,positions,slots,active)]
                out,next1,next2=attention_block(x,cache1,cache2,positions,slots,active,weights,heads,kv,dim,rotary,sections,theta)
            packed=np.concatenate((out,next1.reshape(batch,-1),next2.reshape(batch,-1)),axis=1)
            steps.append(dict(inputs=inputs,expected=packed.ravel().tolist()))
            cache1,cache2=next1.astype(np.float32),next2.astype(np.float32);slots+=active
        with tempfile.TemporaryDirectory(prefix='garnet-pretrained-decoder-') as temporary:
            fixture=dict(config=config,checkpoint=str(checkpoint),steps=steps,memory_only=True)
            path=Path(temporary)/'fixture.json';path.write_text(json.dumps(fixture))
            if len(sys.argv)>4:
                command=[sys.argv[1],str(Path(__file__).with_name('run_cpu_package.py')),str(path),sys.argv[4],sys.argv[2]]
            else:command=[sys.argv[1],sys.argv[2],str(Path(__file__).with_name('run_decoder_graph.py')),str(path)]
            subprocess.run(command,check=True)
            print('native probe cumulative max RSS KiB',resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,flush=True)
        print('pretrained complete decoder',kind,'batch',batch,'PASS',flush=True)
