"""Independent NumPy still-image encoder, FP32 math on decoded weights.

Use explicit einsum for the strided value contraction: on this Mac's NumPy
2.3.4, matmul disagreed with einsum by 1.528 while the identical Windows
fixture agreed within 5.96e-8. GPU output also matched the Windows fixture.
Tolerances and the model equation remain unchanged.
"""
import math
import numpy as np

def reference(patches,weights,c,gh,gw):
    batch,n,_=patches.shape;hidden=c['hidden_size'];heads=c['num_heads'];dim=hidden//heads;merge=c['spatial_merge_size']
    def linear(x,prefix):return x@weights[prefix+'.weight'].T+weights[prefix+'.bias']
    def norm(x,prefix):
        centered=x-np.mean(x,axis=-1,keepdims=True)
        return centered/np.sqrt(np.mean(centered*centered,axis=-1,keepdims=True)+np.float32(1e-6))*weights[prefix+'.weight']+weights[prefix+'.bias']
    def gelu(x,tanh):
        if tanh:return np.float32(.5)*x*(np.float32(1)+np.tanh(np.float32(np.sqrt(2/np.pi))*(x+np.float32(.044715)*x*x*x)))
        return np.float32(.5)*x*(np.float32(1)+np.vectorize(math.erf,otypes=[np.float32])(x/np.sqrt(np.float32(2))))
    x=patches@weights['patch_embed.proj.weight'].reshape(hidden,-1).T+weights['patch_embed.proj.bias']
    table=weights['pos_embed.weight'];side=int(np.sqrt(len(table)))
    coords=np.array([(h,w) for bh in range(0,gh,merge) for bw in range(0,gw,merge)
        for h in range(bh,bh+merge) for w in range(bw,bw+merge)])
    positional=[]
    for h,w in coords:
        ph=np.float32(h*(side-1)/(gh-1)) if gh>1 else np.float32(0)
        pw=np.float32(w*(side-1)/(gw-1)) if gw>1 else np.float32(0)
        h0,w0=int(ph),int(pw);h1,w1=min(h0+1,side-1),min(w0+1,side-1);dh,dw=ph-h0,pw-w0
        positional.append(table[h0*side+w0]*(1-dh)*(1-dw)+table[h0*side+w1]*(1-dh)*dw+
            table[h1*side+w0]*dh*(1-dw)+table[h1*side+w1]*dh*dw)
    x=x+np.asarray(positional,dtype=np.float32)
    theta=c.get('rope_parameters',{}).get('rope_theta',10000)
    freq=theta**(-np.arange(0,dim//2,2,dtype=np.float32)/(dim//2))
    angles=np.concatenate([coords[:,0,None]*freq,coords[:,1,None]*freq],axis=-1)
    angles=np.concatenate([angles,angles],axis=-1).astype(np.float32)
    cos=np.cos(angles)[None,:,None,:];sin=np.sin(angles)[None,:,None,:]
    def rotate(q):return q*cos+np.concatenate([-q[...,dim//2:],q[...,:dim//2]],axis=-1)*sin
    for layer in range(c['depth']):
        prefix='blocks.'+str(layer)
        qkv=linear(norm(x,prefix+'.norm1'),prefix+'.attn.qkv').reshape(batch,n,3,heads,dim)
        q=rotate(qkv[:,:,0]).transpose(0,2,1,3);k=rotate(qkv[:,:,1]).transpose(0,2,1,3);v=qkv[:,:,2].transpose(0,2,1,3)
        scores=np.einsum('bhid,bhjd->bhij',q,k,optimize=False)/np.sqrt(np.float32(dim));prob=np.exp(scores-np.max(scores,axis=-1,keepdims=True));prob/=np.sum(prob,axis=-1,keepdims=True)
        attended=np.einsum('bhij,bhjd->bhid',prob,v, optimize=False).transpose(0,2,1,3).reshape(batch,n,hidden)
        x=x+linear(attended,prefix+'.attn.proj')
        x=x+linear(gelu(linear(norm(x,prefix+'.norm2'),prefix+'.mlp.linear_fc1'),True),prefix+'.mlp.linear_fc2')
    x=norm(x,'merger.norm').reshape(batch,n//(merge*merge),hidden*merge*merge)
    return linear(gelu(linear(x,'merger.linear_fc1'),False),'merger.linear_fc2')

