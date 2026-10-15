"""Independent NumPy Qwen3.5 gated full-attention token reference."""
import numpy as np
def step_attention(x,key,value,positions,slots,active,w,heads,kv,dim,rotary,sections,theta):
    batch=x.shape[0];capacity=key.shape[2]
    projected=(x@w['q'].T).reshape(batch,heads,2*dim)
    q=projected[:,:,:dim];gate=projected[:,:,dim:].reshape(batch,-1)
    k=(x@w['k'].T).reshape(batch,kv,dim);v=(x@w['v'].T).reshape(batch,kv,dim)
    q=q/np.sqrt(np.mean(q*q,axis=-1,keepdims=True)+1e-6)*(1+w['q_norm'])
    k=k/np.sqrt(np.mean(k*k,axis=-1,keepdims=True)+1e-6)*(1+w['k_norm'])
    axes=np.zeros(rotary//2,dtype=np.int64)
    axes[1:sections[1]*3:3]=1;axes[2:sections[2]*3:3]=2
    inverse=(theta**(-2*np.arange(rotary//2,dtype=np.float64)/rotary)).astype(np.float32)
    angles=positions[:,axes]*inverse;angles=np.concatenate((angles,angles),axis=-1)
    cosine=np.cos(angles)[:,None,:];sine=np.sin(angles)[:,None,:]
    def rotate(t):
        part=t[:,:,:rotary]
        halves=np.concatenate((-part[:,:,rotary//2:],part[:,:,:rotary//2]),axis=-1)
        return np.concatenate((part*cosine+halves*sine,t[:,:,rotary:]),axis=-1)
    q=rotate(q);k=rotate(k)
    next_key=key.copy();next_value=value.copy()
    for b in range(batch):
        if active[b]:next_key[b,:,int(slots[b]),:]=k[b];next_value[b,:,int(slots[b]),:]=v[b]
    repeated_key=np.repeat(next_key,heads//kv,axis=1);repeated_value=np.repeat(next_value,heads//kv,axis=1)
    scores=np.einsum('bhd,bhsd->bhs',q,repeated_key)/np.sqrt(dim)
    scores=np.where(np.arange(capacity)[None,None,:]<=slots[:,None,None],scores,-1e30)
    probabilities=np.exp(scores-np.max(scores,axis=-1,keepdims=True))
    probabilities/=np.sum(probabilities,axis=-1,keepdims=True)
    context=np.einsum('bhs,bhsd->bhd',probabilities,repeated_value).reshape(batch,-1)
    output=(context/(1+np.exp(-gate)))@w['out'].T;output[active==0]=0
    return output,next_key,next_value
