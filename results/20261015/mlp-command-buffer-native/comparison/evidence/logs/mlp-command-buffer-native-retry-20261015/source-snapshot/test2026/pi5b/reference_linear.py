"""Independent NumPy reference for one Qwen3.5 GatedDeltaNet token."""
import numpy as np
def step_linear(x,conv_state,state,active,w,kh,vh,kd,vd):
    batch=x.shape[0]
    projected=x@w['qkv'].T
    next_conv=np.concatenate((conv_state[:,:,1:],projected[:,:,None]),axis=-1)
    convolution=np.sum(next_conv*w['conv'][:,0,:],axis=-1)
    mixed=convolution/(1+np.exp(-convolution))
    q,k,v=np.split(mixed,[kh*kd,2*kh*kd],axis=-1)
    q=np.repeat(q.reshape(batch,kh,kd),vh//kh,axis=1)
    k=np.repeat(k.reshape(batch,kh,kd),vh//kh,axis=1)
    v=v.reshape(batch,vh,vd)
    q=q/np.sqrt(np.sum(q*q,axis=-1,keepdims=True)+1e-6)/np.sqrt(kd)
    k=k/np.sqrt(np.sum(k*k,axis=-1,keepdims=True)+1e-6)
    beta=1/(1+np.exp(-(x@w['b_proj'].T)))
    g=-np.exp(w['a_log'])*np.logaddexp(0,x@w['a_proj'].T+w['dt_bias'])
    next_state=state*np.exp(g)[...,None,None]
    prediction=np.einsum('bhkv,bhk->bhv',next_state,k)
    next_state+=k[...,None]*((v-prediction)*beta[...,None])[...,None,:]
    core=np.einsum('bhkv,bhk->bhv',next_state,q)
    normalized=core/np.sqrt(np.mean(core*core,axis=-1,keepdims=True)+1e-6)*w['norm']
    z=(x@w['z'].T).reshape(batch,vh,vd)
    gated=normalized*z/(1+np.exp(-z))
    out=gated.reshape(batch,vh*vd)@w['out_proj'].T
    next_state[active==0]=state[active==0];next_conv[active==0]=conv_state[active==0];out[active==0]=0
    return out,next_conv,next_state
