"""Independent NumPy checks for real mixed-precision GDN/full decoder layers."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from prepare_real_mlp_probe import Checkpoint,bf16,fp8_table,nearest,project as packed_project
FP8_ALPHA_AFTER_DOT=False

def dense(cp,name):
    raw,e=cp.tensor(name)
    if e['dtype']=='BF16':out=(raw.view('<u2').astype(np.uint32)<<16).view(np.float32)
    elif e['dtype']=='F32':out=raw.view('<f4')
    else:raise ValueError('Expected BF16/F32 dense tensor '+name)
    return out.reshape(e['shape'])

def project(cp,prefix,x):
    raw,e=cp.tensor(prefix+'.weight')
    if e['dtype']=='U8':return packed_project(cp,prefix,x)
    n,k=e['shape'];x=bf16(x)
    if e['dtype']=='BF16':return bf16(dense(cp,prefix+'.weight')@x)
    if e['dtype'] not in ('F8_E4M3','F8_E4M3FN'):raise ValueError('Unsupported projection')
    table=fp8_table();sx=cp.scalar(prefix+'.input_scale');sw=cp.scalar(prefix+'.weight_scale')
    normalized=x/sx;q=table[nearest(np.abs(normalized),table)]*np.where(np.signbit(normalized),-1,1)
    activation=(q*sx).astype(np.float32);out=np.empty(n,dtype=np.float32);raw=raw.reshape(n,k)
    for start in range(0,n,128):
        codes=raw[start:start+128]
        if np.any((codes&127)==127):raise ValueError('NaN FP8 weight')
        w=table[codes&127]*np.where(codes&128,-1,1)
        if FP8_ALPHA_AFTER_DOT:
            alpha=np.float32(np.float32(sw)*np.float32(sx))
            out[start:start+128]=(w.astype(np.float32)@q.astype(np.float32))*alpha
        else:out[start:start+128]=((w*sw).astype(np.float32)@activation)
    return bf16(out)

def rms(x,w,eps,offset=False):
    return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+np.float32(eps))*(w+np.float32(offset))
def silu(x):return x/(1+np.exp(-x))
def mlp(cp,x,p):return project(cp,p+'.down_proj',silu(project(cp,p+'.gate_proj',x))*project(cp,p+'.up_proj',x))

def linear(cp,x,history,state,p,t):
    kh,vh,kd,vd=[t[k] for k in ('linear_num_key_heads','linear_num_value_heads','linear_key_head_dim','linear_value_head_dim')]
    qkv=project(cp,p+'.in_proj_qkv',x);z=project(cp,p+'.in_proj_z',x).reshape(vh,vd)
    beta=1/(1+np.exp(-project(cp,p+'.in_proj_b',x)))
    a=project(cp,p+'.in_proj_a',x)+dense(cp,p+'.dt_bias')
    decay=np.exp(-np.exp(dense(cp,p+'.A_log'))*np.logaddexp(np.float32(0),a))
    history=np.concatenate([history[:,1:],qkv[:,None]],axis=1)
    conv=silu(np.sum(history*dense(cp,p+'.conv1d.weight').reshape(history.shape),axis=1))
    q,k,v=np.split(conv,[kh*kd,2*kh*kd]);q=np.repeat(q.reshape(kh,kd),vh//kh,axis=0);k=np.repeat(k.reshape(kh,kd),vh//kh,axis=0);v=v.reshape(vh,vd)
    q=q/np.sqrt(np.sum(q*q,axis=1,keepdims=True)+np.float32(1e-6))/np.float32(np.sqrt(kd))
    k=k/np.sqrt(np.sum(k*k,axis=1,keepdims=True)+np.float32(1e-6))
    state=state*decay[:,None,None];delta=(v-np.sum(state*k[:,:,None],axis=1))*beta[:,None]
    state=state+k[:,:,None]*delta[:,None,:]
    output=np.sum(state*q[:,:,None],axis=1)
    output=rms(output,dense(cp,p+'.norm.weight'),t['rms_norm_eps'])*silu(z)
    return project(cp,p+'.out_proj',output.reshape(-1)),history,state

def attention(cp,x,key,value,p,t,position,slot):
    heads,kv,d=[t[k] for k in ('num_attention_heads','num_key_value_heads','head_dim')];eps=t['rms_norm_eps']
    qg=project(cp,p+'.q_proj',x).reshape(heads,2*d)
    q=rms(qg[:,:d],dense(cp,p+'.q_norm.weight'),eps,True)
    k=rms(project(cp,p+'.k_proj',x).reshape(kv,d),dense(cp,p+'.k_norm.weight'),eps,True)
    v=project(cp,p+'.v_proj',x).reshape(kv,d)
    rope=t['rope_parameters'];rot=int(d*rope['partial_rotary_factor']);sections=rope['mrope_section']
    freq=np.array([rope['rope_theta']**(-2*i/rot) for i in range(rot//2)],dtype=np.float32)
    axes=[1 if i%3==1 and i<sections[1]*3 else (2 if i%3==2 and i<sections[2]*3 else 0) for i in range(rot//2)]
    angle=np.tile(np.asarray(position,dtype=np.float32)[axes]*freq,2);cos=np.cos(angle);sin=np.sin(angle)
    def rotate(a):
        part=a[:,:rot];half=np.concatenate([-part[:,rot//2:],part[:,:rot//2]],axis=1)
        return np.concatenate([part*cos+half*sin,a[:,rot:]],axis=1)
    q=rotate(q);key=key.copy();value=value.copy();key[:,slot]=rotate(k);value[:,slot]=v
    keys=np.repeat(key[:,:slot+1],heads//kv,axis=0);values=np.repeat(value[:,:slot+1],heads//kv,axis=0)
    scores=np.sum(q[:,None,:]*keys,axis=-1)/np.float32(np.sqrt(d));prob=np.exp(scores-np.max(scores,axis=1,keepdims=True));prob/=np.sum(prob,axis=1,keepdims=True)
    output=np.sum(prob[:,:,None]*values,axis=1)/(1+np.exp(-qg[:,d:]))
    return project(cp,p+'.o_proj',output.reshape(-1)),key,value

def prepare(root,capacity=4096):
    cp=Checkpoint(root);config=json.loads((root/'config.json').read_text());t=config['text_config'];width=t['hidden_size'];layers=[]
    for layer in [0,3]:
        kind=t['layer_types'][layer];p='model.language_model.layers.'+str(layer)
        if kind=='linear_attention':
            channels=2*t['linear_num_key_heads']*t['linear_key_head_dim']+t['linear_num_value_heads']*t['linear_value_head_dim']
            s1=np.zeros((channels,t['linear_conv_kernel_dim']),dtype=np.float32);s2=np.zeros((t['linear_num_value_heads'],t['linear_key_head_dim'],t['linear_value_head_dim']),dtype=np.float32)
        else:s1=np.zeros((t['num_key_value_heads'],capacity,t['head_dim']),dtype=np.float32);s2=s1.copy()
        cases=[]
        for step in range(2):
            x=np.array([((i*17+step*23)%101-50)/50 for i in range(width)],dtype=np.float32);position=[3+step,5+step,7+step]
            normalized=rms(x,dense(cp,p+'.input_layernorm.weight'),t['rms_norm_eps'],True)
            if kind=='linear_attention':a,s1,s2=linear(cp,normalized,s1,s2,p+'.linear_attn',t)
            else:a,s1,s2=attention(cp,normalized,s1,s2,p+'.self_attn',t,position,step)
            residual=x+a;out=residual+mlp(cp,rms(residual,dense(cp,p+'.post_attention_layernorm.weight'),t['rms_norm_eps'],True),p+'.mlp')
            samples=[]
            for state in (s1,s2):
                flat=state.reshape(-1);indices=set([0,len(flat)-1]+[int(i) for i in np.linspace(0,len(flat)-1,512,dtype=int)])
                if kind=='full_attention':
                    for head in range(t['num_key_value_heads']):
                        for slot in range(step+1):
                            for col in np.linspace(0,t['head_dim']-1,64,dtype=int):indices.add((head*capacity+slot)*t['head_dim']+int(col))
                indices=sorted(indices)
                samples.append({'indices':indices,'values':flat[indices].tolist()})
            cases.append({'input':x.tolist(),'positions':position,'expected_hidden':out.tolist(),'state_samples':samples})
        layers.append({'layer':layer,'kind':kind,'state_shapes':[list(s1.shape),list(s2.shape)],'cases':cases})
    return {'layers':layers,'capacity':capacity,'numpy_version':np.__version__,
        'scope':'Actual decoder layers0/3, two stateful tokens each; all hidden values, uniform state samples and active KV-prefix samples. Independent CPU arithmetic, not full-model quality or performance.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    result=prepare(a.checkpoint);a.output.write_text(json.dumps(result))
    print(json.dumps({'oracle_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'scope':result['scope']}))
