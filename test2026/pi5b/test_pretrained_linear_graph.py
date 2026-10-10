"""Execute layer 0 of the pinned pretrained checkpoint through actual Garnet.

This is a full-size layer check, not text generation or vision inference.
Reference uses FP32 arithmetic with decoded official BF16/F32 tensor values.
"""
import json,resource,struct,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from reference_linear import step_linear
checkpoint=Path(sys.argv[3]).resolve()
raw=checkpoint.read_bytes();header_length=struct.unpack('<Q',raw[:8])[0]
header=json.loads(raw[8:8+header_length]);data_start=8+header_length
suffixes=dict(qkv='in_proj_qkv.weight',z='in_proj_z.weight',b_proj='in_proj_b.weight',a_proj='in_proj_a.weight',
    conv='conv1d.weight',a_log='A_log',dt_bias='dt_bias',norm='norm.weight',out_proj='out_proj.weight')
weights={}
for name,suffix in suffixes.items():
    entry=header['model.language_model.layers.0.linear_attn.'+suffix]
    a,b=entry['data_offsets'];data=raw[data_start+a:data_start+b]
    if entry['dtype']=='BF16':array=(np.frombuffer(data,dtype='<u2').astype(np.uint32)<<16).view(np.float32)
    elif entry['dtype']=='F32':array=np.frombuffer(data,dtype='<f4')
    else:raise ValueError('unexpected pretrained dtype')
    weights[name]=array.reshape(entry['shape'])
rng=np.random.default_rng(532)
kh=vh=16;kd=vd=128;d=1024;ks=4;channels=6144
for batch in (1,2):
    config=dict(batch=batch,mode='packed',hidden_size=d,linear_num_key_heads=kh,linear_num_value_heads=vh,
        linear_key_head_dim=kd,linear_value_head_dim=vd,linear_conv_kernel_dim=ks,rms_norm_eps=1e-6)
    state=np.zeros((batch,vh,kd,vd),dtype=np.float32);conv=np.zeros((batch,channels,ks),dtype=np.float32)
    steps=[]
    for step in range(4):
        x=rng.normal(size=(batch,d)).astype(np.float32);active=np.ones(batch,dtype=np.float32)
        if batch==2 and step==2:active[1]=0
        inputs=[x.ravel().tolist(),conv.ravel().tolist(),state.ravel().tolist(),active.tolist()]
        out,next_conv,next_state=step_linear(x,conv,state,active,weights,kh,vh,kd,vd)
        packed=np.concatenate((out,next_conv.reshape(batch,-1),next_state.reshape(batch,-1)),axis=1)
        steps.append(dict(inputs=inputs,expected=packed.ravel().tolist()))
        conv,state=next_conv.astype(np.float32),next_state.astype(np.float32)
    with tempfile.TemporaryDirectory(prefix='garnet-pretrained-linear-') as folder:
        fixture=dict(config=config,checkpoint=str(checkpoint),steps=steps,rtol=5e-5,atol=5e-6)
        fixture_path=Path(folder)/'fixture.json';fixture_path.write_text(json.dumps(fixture))
        subprocess.run([sys.argv[1],sys.argv[2],str(Path(__file__).with_name('run_linear_graph.py')),str(fixture_path)],check=True)
        print('native probe cumulative max RSS KiB',resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,flush=True)
    print('pretrained full-size captured linear layer batch',batch,'PASS',flush=True)
