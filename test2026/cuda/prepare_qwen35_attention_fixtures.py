"""Independent full-attention parity fixtures for CUDA, including MRoPE and inactive rows."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
from qwen35_step_reference import attention
root=Path(sys.argv[1]).resolve();root.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('reference',Path(__file__).resolve().parents[1]/'pi5b/reference_attention.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
rng=np.random.default_rng(933)
partial=len(sys.argv)>2 and sys.argv[2]=='partial'
real=len(sys.argv)>2 and sys.argv[2]=='real'
hidden,heads,kv,dim,rotary,capacity=8,4,2,32 if partial else 8,8,8
sections=[2,1,1];theta=10000.
if real:
    heads,kv,dim,rotary=16,4,256,64
    sections=[11,11,10];theta=10000000.
w0={'q':rng.normal(0,.2,(2*heads*dim,hidden)),'k':rng.normal(0,.2,(kv*dim,hidden)),
 'v':rng.normal(0,.2,(kv*dim,hidden)),'out':rng.normal(0,.2,(hidden,heads*dim)),
 'q_norm':rng.normal(0,.1,dim),'k_norm':rng.normal(0,.1,dim)}
names={'q':'q_proj.weight','k':'k_proj.weight','v':'v_proj.weight','out':'o_proj.weight','q_norm':'q_norm.weight','k_norm':'k_norm.weight'}
cases=[]
for dtype in ['float32','bfloat16']:
    checkpoint=root/dtype;checkpoint.mkdir(exist_ok=True)
    ts={k:torch.from_numpy(np.asarray(v,dtype=np.float32)).to(getattr(torch,dtype)) for k,v in w0.items()}
    save_file({'model.language_model.layers.0.self_attn.'+names[k]:v for k,v in ts.items()},str(checkpoint/'model.safetensors'))
    w={k:v.float().numpy() for k,v in ts.items()}
    for batch in ([2] if partial else [1,2,4,8]):
        key=rng.normal(0,.03,(batch,kv,capacity,dim)).astype(np.float32);value=key.copy();steps=[]
        for step in range(3):
            x=rng.normal(0,.2,(batch,hidden)).astype(np.float32)
            active=np.array([float((b+step)%2==0) for b in range(batch)],dtype=np.float32)
            positions=np.array([[step+b,step+2*b,step+3*b] for b in range(batch)],dtype=np.float32)
            slots=np.full(batch,step,dtype=np.float32)
            out,nk,nv=attention(x,key,value,positions,slots,active,ts,heads,kv,dim,rotary,sections,theta)
            if dtype=='float32':
                independent=ref.step_attention(x,key,value,positions,slots,active,w,heads,kv,dim,rotary,sections,theta)
                for a,e in zip([out,nk,nv],independent):np.testing.assert_allclose(a,e,rtol=1e-4,atol=1e-6)
            packed=np.concatenate([out,nk.reshape(batch,-1),nv.reshape(batch,-1)],axis=1).astype(np.float32)
            arrays=[x,key,value,positions,slots,active]
            steps.append({'inputs':[{'shape':list(a.shape),'values':a.ravel().tolist()} for a in arrays],
                'expected':packed.ravel().tolist(),'output_shape':list(packed.shape)})
            key,value=nk,nv
        cases.append({'kind':'attention','dtype':dtype,'batch':batch,'checkpoint':str(checkpoint),
            'config':{'hidden_size':hidden,'num_attention_heads':heads,'num_key_value_heads':kv,'head_dim':dim,'rms_norm_eps':1e-6,
                'rope_parameters':{'partial_rotary_factor':rotary/dim,'rope_theta':theta,'mrope_section':sections}},'steps':steps})
(root/'fixtures.json').write_text(json.dumps(cases))
print('Prepared',len(cases),'independent full-attention cases')
