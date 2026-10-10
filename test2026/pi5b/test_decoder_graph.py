import json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from checkpoint_fixture import write_checkpoint
from reference_decoder import linear_block,attention_block
rng=np.random.default_rng(535)
common=dict(input_norm='input_layernorm.weight',post_norm='post_attention_layernorm.weight',
    mlp_gate='mlp.gate_proj.weight',mlp_up='mlp.up_proj.weight',mlp_down='mlp.down_proj.weight')
for kind in ('linear','full'):
    for batch in (1,2):
        d,mid=11,17;weights=dict(input_norm=rng.normal(0,.2,d).astype(np.float32),post_norm=rng.normal(0,.2,d).astype(np.float32),
            mlp_gate=rng.normal(0,.2,(mid,d)).astype(np.float32),mlp_up=rng.normal(0,.2,(mid,d)).astype(np.float32),mlp_down=rng.normal(0,.2,(d,mid)).astype(np.float32))
        config=dict(kind=kind,batch=batch,hidden_size=d,intermediate_size=mid,rms_norm_eps=1e-6)
        suffixes=common.copy()
        if kind=='linear':
            kh,vh,kd,vd,ks=1,2,7,5,4;channels=2*kh*kd+vh*vd;layer=0
            shapes=dict(qkv=(channels,d),z=(vh*vd,d),b_proj=(vh,d),a_proj=(vh,d),conv=(channels,1,ks),a_log=(vh,),dt_bias=(vh,),norm=(vd,),out_proj=(d,vh*vd))
            names=dict(qkv='in_proj_qkv.weight',z='in_proj_z.weight',b_proj='in_proj_b.weight',a_proj='in_proj_a.weight',
                conv='conv1d.weight',a_log='A_log',dt_bias='dt_bias',norm='norm.weight',out_proj='out_proj.weight')
            suffixes.update({name:'linear_attn.'+suffix for name,suffix in names.items()})
            config.update(linear_num_key_heads=kh,linear_num_value_heads=vh,linear_key_head_dim=kd,linear_value_head_dim=vd,linear_conv_kernel_dim=ks)
            cache1=rng.normal(0,.2,(batch,channels,ks)).astype(np.float32);cache2=rng.normal(0,.2,(batch,vh,kd,vd)).astype(np.float32)
        else:
            heads,kv,dim,rotary,capacity=4,2,12,8,8;layer=3;sections=[2,1,1];theta=10000000
            shapes=dict(q=(2*heads*dim,d),k=(kv*dim,d),v=(kv*dim,d),out=(d,heads*dim),q_norm=(dim,),k_norm=(dim,))
            names=dict(q='q_proj.weight',k='k_proj.weight',v='v_proj.weight',out='o_proj.weight',q_norm='q_norm.weight',k_norm='k_norm.weight')
            suffixes.update({name:'self_attn.'+suffix for name,suffix in names.items()})
            config.update(num_attention_heads=heads,num_key_value_heads=kv,head_dim=dim,capacity=capacity,
                rope_parameters=dict(partial_rotary_factor=rotary/dim,rope_theta=theta,mrope_section=sections))
            cache1=rng.normal(0,.2,(batch,kv,capacity,dim)).astype(np.float32);cache2=rng.normal(0,.2,cache1.shape).astype(np.float32)
        config['layer_idx']=layer
        weights.update({name:rng.normal(0,.2,shape).astype(np.float32) for name,shape in shapes.items()})
        if kind=='linear':weights['norm']+=1
        with tempfile.TemporaryDirectory(prefix='garnet-decoder-') as folder:
            checkpoint=Path(folder)/'weights.safetensors'
            weights=write_checkpoint(checkpoint,weights,suffixes,'model.language_model.layers.'+str(layer),('a_log','norm'))
            steps=[];slots=np.zeros(batch,dtype=np.float32)
            for step in range(6):
                x=rng.normal(size=(batch,d)).astype(np.float32);active=np.ones(batch,dtype=np.float32)
                if batch==2 and 2<=step<=3:active[1]=0
                if kind=='linear':
                    inputs=[a.ravel().tolist() for a in (x,cache1,cache2,active)]
                    output,next1,next2=linear_block(x,cache1,cache2,active,weights,kh,vh,kd,vd)
                else:
                    positions=np.stack((slots*2+1,slots*3+4,slots*5+7),axis=-1).astype(np.float32)
                    inputs=[a.ravel().tolist() for a in (x,cache1,cache2,positions,slots,active)]
                    output,next1,next2=attention_block(x,cache1,cache2,positions,slots,active,weights,heads,kv,dim,rotary,sections,theta)
                packed=np.concatenate((output,next1.reshape(batch,-1),next2.reshape(batch,-1)),axis=1)
                steps.append(dict(inputs=inputs,expected=packed.ravel().tolist()))
                cache1,cache2=next1.astype(np.float32),next2.astype(np.float32);slots+=active
            fixture=dict(config=config,checkpoint=str(checkpoint),steps=steps,memory_only=True)
            path=Path(folder)/'fixture.json';path.write_text(json.dumps(fixture))
            if len(sys.argv)>3:
                command=[sys.argv[1],str(Path(__file__).with_name('run_cpu_package.py')),str(path),sys.argv[3],sys.argv[2]]
            else:
                command=[sys.argv[1],sys.argv[2],str(Path(__file__).with_name('run_decoder_graph.py')),str(path)]
            subprocess.run(command,check=True)
        print('complete captured decoder block',kind,'batch',batch,'PASS',flush=True)
