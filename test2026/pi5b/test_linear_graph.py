"""Numerically execute actual xModel capture, weight mapping, and state slices."""
import json,struct,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from reference_linear import step_linear
rng=np.random.default_rng(531)
suffixes=dict(qkv='in_proj_qkv.weight',z='in_proj_z.weight',b_proj='in_proj_b.weight',a_proj='in_proj_a.weight',
    conv='conv1d.weight',a_log='A_log',dt_bias='dt_bias',norm='norm.weight',out_proj='out_proj.weight')
for batch in (1,2):
    for kh,vh in ((2,2),(1,2)):
        kd,vd,d,ks=7,5,11,4;channels=2*kh*kd+vh*vd
        shapes=dict(qkv=(channels,d),z=(vh*vd,d),b_proj=(vh,d),a_proj=(vh,d),conv=(channels,1,ks),a_log=(vh,),dt_bias=(vh,),norm=(vd,),out_proj=(d,vh*vd))
        weights={name:rng.normal(0,.2,size=shape).astype(np.float32) for name,shape in shapes.items()}
        weights['norm']+=1
        # The real checkpoint mixes BF16 projections and FP32 scalar/norm weights.
        header={};payload=bytearray()
        for name,weight in weights.items():
            dtype='F32' if name in ('a_log','dt_bias','norm') else 'BF16'
            if dtype=='BF16':
                bits=weight.view(np.uint32)
                bf16=((bits+0x7fff+((bits>>16)&1))>>16).astype('<u2')
                weights[name]=(bf16.astype(np.uint32)<<16).view(np.float32)
                raw=bf16.tobytes()
            else:raw=weight.astype('<f4').tobytes()
            key='model.language_model.layers.0.linear_attn.'+suffixes[name]
            header[key]=dict(dtype=dtype,shape=list(weight.shape),data_offsets=[len(payload),len(payload)+len(raw)])
            payload.extend(raw)
        encoded=json.dumps(header,separators=(',',':')).encode()
        encoded+=b' '*((-len(encoded))%8)
        config=dict(batch=batch,hidden_size=d,linear_num_key_heads=kh,linear_num_value_heads=vh,
            linear_key_head_dim=kd,linear_value_head_dim=vd,linear_conv_kernel_dim=ks,rms_norm_eps=1e-6)
        state=rng.normal(0,.2,size=(batch,vh,kd,vd)).astype(np.float32)
        conv=rng.normal(0,.2,size=(batch,channels,ks)).astype(np.float32)
        steps=[]
        for step in range(7):
            x=rng.normal(size=(batch,d)).astype(np.float32);active=np.ones(batch,dtype=np.float32)
            if batch==2 and 2<=step<=4:active[1]=0
            inputs=[x.ravel().tolist(),conv.ravel().tolist(),state.ravel().tolist(),active.tolist()]
            out,next_conv,next_state=step_linear(x,conv,state,active,weights,kh,vh,kd,vd)
            packed=np.concatenate((out,next_conv.reshape(batch,-1),next_state.reshape(batch,-1)),axis=1)
            steps.append(dict(inputs=inputs,outputs=dict(out=out.ravel().tolist(),conv_state=next_conv.ravel().tolist(),
                state=next_state.ravel().tolist(),packed=packed.ravel().tolist())))
            conv,state=next_conv.astype(np.float32),next_state.astype(np.float32)
        with tempfile.TemporaryDirectory(prefix='garnet-linear-') as folder:
            checkpoint=Path(folder)/'weights.safetensors'
            checkpoint.write_bytes(struct.pack('<Q',len(encoded))+encoded+payload)
            for mode in ('packed','out','conv_state','state'):
                fixture=dict(config=dict(config,mode=mode),checkpoint=str(checkpoint),steps=[dict(inputs=s['inputs'],expected=s['outputs'][mode]) for s in steps])
                fixture_path=Path(folder)/'fixture.json';fixture_path.write_text(json.dumps(fixture))
                subprocess.run([sys.argv[1],sys.argv[2],str(Path(__file__).with_name('run_linear_graph.py')),str(fixture_path)],check=True)
        print('captured BF16/F32 checkpoint linear graph and extraction batch',batch,'heads',kh,vh,'PASS',flush=True)
