"""Independent PyTorch reference for the declared mixed-precision CUDA steps.

Projection inputs/outputs round to checkpoint dtype. State, norms, RoPE and
attention reductions remain FP32. Full pretrained-model agreement is a
separate validation requirement from this component precision contract.
"""
import math
import torch
import torch.nn.functional as F

def attention(x,key,value,positions,slots,active,w,heads,kv,dim,rotary,sections,theta):
    x,key,value,positions,slots,active=[torch.from_numpy(a.copy()) for a in [x,key,value,positions,slots,active]]
    def project(a,name):return F.linear(a.to(w[name].dtype),w[name]).float()
    batch=x.shape[0];capacity=key.shape[2]
    queryGate=project(x,'q').reshape(batch,heads,2*dim)
    q,gate=queryGate.chunk(2,-1);gate=gate.reshape(batch,-1)
    k=project(x,'k').reshape(batch,kv,dim);v=project(x,'v').reshape(batch,kv,dim)
    q=q*torch.rsqrt(q.square().mean(-1,keepdim=True)+1e-6)*(1+w['q_norm'].float())
    k=k*torch.rsqrt(k.square().mean(-1,keepdim=True)+1e-6)*(1+w['k_norm'].float())
    axes=torch.zeros(rotary//2,dtype=torch.long)
    axes[1:sections[1]*3:3]=1;axes[2:sections[2]*3:3]=2
    frequencies=torch.tensor([theta**(-2*i/rotary) for i in range(rotary//2)],dtype=torch.float32)
    angle=positions[:,axes]*frequencies;angle=torch.cat([angle,angle],-1).unsqueeze(1)
    def rotate(a):
        part=a[...,:rotary];left,right=part.chunk(2,-1)
        result=part*angle.cos()+torch.cat([-right,left],-1)*angle.sin()
        return torch.cat([result,a[...,rotary:]],-1)
    q,k=rotate(q),rotate(k)
    nextKey,nextValue=key.clone(),value.clone()
    for b in range(batch):
        if active[b]:nextKey[b,:,int(slots[b])]=k[b];nextValue[b,:,int(slots[b])]=v[b]
    rk=nextKey.repeat_interleave(heads//kv,1);rv=nextValue.repeat_interleave(heads//kv,1)
    scores=torch.einsum('bhd,bhsd->bhs',q,rk)/math.sqrt(dim)
    scores=scores.masked_fill(torch.arange(capacity)[None,None,:]>slots[:,None,None],-1e30)
    context=torch.einsum('bhs,bhsd->bhd',scores.softmax(-1),rv).reshape(batch,-1)
    out=project(context*gate.sigmoid(),'out');out[active==0]=0
    return [a.numpy() for a in [out,nextKey,nextValue]]

def linear(x,history,state,active,w,kh,vh,kd,vd):
    x,history,state,active=[torch.from_numpy(a.copy()) for a in [x,history,state,active]]
    def project(name):return F.linear(x.to(w[name].dtype),w[name]).float()
    batch=x.shape[0]
    nextHistory=torch.cat([history[:,:,1:],project('qkv').unsqueeze(-1)],-1)
    mixed=F.silu((nextHistory*w['conv'].float()[:,0]).sum(-1))
    q,k,v=torch.split(mixed,[kh*kd,kh*kd,vh*vd],-1)
    q=q.reshape(batch,kh,kd).repeat_interleave(vh//kh,1)
    k=k.reshape(batch,kh,kd).repeat_interleave(vh//kh,1);v=v.reshape(batch,vh,vd)
    q=q*torch.rsqrt(q.square().sum(-1,keepdim=True)+1e-6)/math.sqrt(kd)
    k=k*torch.rsqrt(k.square().sum(-1,keepdim=True)+1e-6)
    beta=project('b_proj').sigmoid()
    gate=-w['a_log'].float().exp()*F.softplus(project('a_proj')+w['dt_bias'].float())
    nextState=state*gate.exp()[...,None,None]
    memory=torch.einsum('bhkv,bhk->bhv',nextState,k)
    nextState=nextState+k[...,None]*((v-memory)*beta[...,None])[...,None,:]
    core=torch.einsum('bhkv,bhk->bhv',nextState,q)
    core=core*torch.rsqrt(core.square().mean(-1,keepdim=True)+1e-6)*w['norm'].float()
    z=F.silu(project('z').reshape(batch,vh,vd));mixed=(core*z).reshape(batch,vh*vd)
    out=F.linear(mixed.to(w['out_proj'].dtype),w['out_proj']).float()
    nextHistory[active==0]=history[active==0];nextState[active==0]=state[active==0];out[active==0]=0
    return [a.numpy() for a in [out,nextHistory,nextState]]
