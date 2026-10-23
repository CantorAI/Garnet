"""Independent NumPy recurrent-state reference, including grouped heads/batches.

Run with CPython + NumPy/safetensors/torch; run the generated fixtures with
test_qwen35_linear.py in the native XLang3 runtime. This is component parity,
not a pretrained-model or performance benchmark.
"""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
from qwen35_step_reference import linear

root=Path(sys.argv[1]).resolve();root.mkdir(parents=True,exist_ok=True)
reference=Path(__file__).resolve().parents[1]/'pi5b/reference_linear.py'
spec=importlib.util.spec_from_file_location('reference',reference)
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
rng=np.random.default_rng(932)
hidden,kh,vh,kd,vd,kernel=8,2,4,4,4,4
channels=2*kh*kd+vh*vd
weights={'qkv':rng.normal(0,.15,(channels,hidden)),'z':rng.normal(0,.1,(vh*vd,hidden)),
 'b_proj':rng.normal(0,.1,(vh,hidden)),'a_proj':rng.normal(0,.1,(vh,hidden)),
 'conv':rng.normal(0,.2,(channels,1,kernel)),'a_log':rng.normal(0,.1,vh),
 'dt_bias':np.array([-30.,0.,30.,100.]),'norm':rng.uniform(.8,1.2,vd),
 'out_proj':rng.normal(0,.1,(hidden,vh*vd))}
names={'qkv':'in_proj_qkv.weight','z':'in_proj_z.weight','b_proj':'in_proj_b.weight',
 'a_proj':'in_proj_a.weight','conv':'conv1d.weight','a_log':'A_log','dt_bias':'dt_bias',
 'norm':'norm.weight','out_proj':'out_proj.weight'}
cases=[]
for dtype in ['float32','bfloat16']:
    checkpoint=root/dtype;checkpoint.mkdir(exist_ok=True)
    tensors={k:torch.from_numpy(np.asarray(v,dtype=np.float32)).to(getattr(torch,dtype)) for k,v in weights.items()}
    save_file({'model.language_model.layers.0.linear_attn.'+names[k]:v for k,v in tensors.items()},str(checkpoint/'model.safetensors'))
    w={k:v.float().numpy() for k,v in tensors.items()}
    for batch in [1,2,4,8]:
        conv=rng.normal(0,.03,(batch,channels,kernel)).astype(np.float32)
        state=rng.normal(0,.02,(batch,vh,kd,vd)).astype(np.float32)
        steps=[]
        for step in range(3):
            x=rng.normal(0,.15,(batch,hidden)).astype(np.float32)
            active=np.array([float((b+step)%2==0) for b in range(batch)],dtype=np.float32)
            output,nextConv,nextState=linear(x,conv,state,active,tensors,kh,vh,kd,vd)
            if dtype=='float32':
                independent=ref.step_linear(x,conv,state,active,w,kh,vh,kd,vd)
                for a,e in zip([output,nextConv,nextState],independent):np.testing.assert_allclose(a,e,rtol=1e-4,atol=1e-6)
            packed=np.concatenate([output,nextConv.reshape(batch,-1),nextState.reshape(batch,-1)],axis=1).astype(np.float32)
            arrays=[x,conv,state,active]
            steps.append({'inputs':[{'shape':list(a.shape),'values':a.ravel().tolist()} for a in arrays],
                'expected':packed.ravel().tolist(),'output_shape':list(packed.shape)})
            conv,state=nextConv,nextState
        cases.append({'dtype':dtype,'batch':batch,'checkpoint':str(checkpoint),
            'config':{'hidden_size':hidden,'linear_num_key_heads':kh,'linear_num_value_heads':vh,
                'linear_key_head_dim':kd,'linear_value_head_dim':vd,'linear_conv_kernel_dim':kernel,'rms_norm_eps':1e-6},'steps':steps})
(root/'fixtures.json').write_text(json.dumps(cases))
print('Prepared',len(cases),'independent cases with three explicit recurrent-state steps each')
