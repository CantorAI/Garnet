"""Independent FP32 recurrence versus triangular-solve chunk formulation.

CPU algebra/numerical experiment only: no Metal or serving speed claim.
State layout is [..., key, value], matching Garnet rather than MLX.
Decay is supplied as log-decay. No clipping or division by decay is used.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np

def sequential(q,k,v,log_decay,beta,state,active):
 s=state.copy();ys=[]
 for t in range(q.shape[-2]):
  mask=active[...,t,None,None]
  decay=np.exp(log_decay[...,t],dtype=np.float32)[...,None,None]
  decayed=s*decay
  memory=np.sum(k[...,t,:,None]*decayed,axis=-2,dtype=np.float32)
  change=(v[...,t,:]-memory)*beta[...,t,None]
  updated=decayed+k[...,t,:,None]*change[...,None,:]
  s=np.where(mask,updated,s)
  ys.append(np.where(active[...,t,None],np.sum(q[...,t,:,None]*s,axis=-2,dtype=np.float32),0))
 return np.stack(ys,axis=-2),s

def matmul(a,b):
 # Explicit FP32 contractions avoid the unstable batched-matmul CPU path
 # observed in the real-weight diagnostic. This is not a Metal kernel.
 return np.einsum('...ik,...kj->...ij',a,b,optimize=False,dtype=np.float32)

def chunked(q, k, v, log_decay, beta, state, active):
    n = q.shape[-2]
    g = np.where(active, np.exp(log_decay, dtype=np.float32), np.float32(1))
    b = np.where(active, beta, np.float32(0))
    dr = np.zeros(g.shape + (n,), np.float32)
    for i in range(n):
        dr[..., i, i] = 1
        if i:
            dr[..., i, :i] = dr[..., i - 1, :i] * g[..., i, None]
    cumulative = np.cumprod(g, axis=-1, dtype=np.float32)
    gram = matmul(k, np.swapaxes(k, -1, -2))
    a = b[..., None] * dr * gram
    rhs = b[..., None] * (v - cumulative[..., None] * matmul(k, state))
    u = np.empty_like(rhs)
    for i in range(n):
        u[..., i, :] = rhs[..., i, :]
        if i:
            u[..., i, :] -= matmul(a[..., i:i + 1, :i], u[..., :i, :])[..., 0, :]
    y = cumulative[..., None] * matmul(q, state) + matmul(dr * matmul(q, np.swapaxes(k, -1, -2)), u)
    s = cumulative[..., -1, None, None] * state + matmul(np.swapaxes(k, -1, -2), dr[..., -1, :, None] * u)
    return (np.where(active[..., None], y, np.float32(0)), s)

def segmented(q,k,v,g,b,state,active,size):
 s=state.copy();ys=[]
 for begin in range(0,q.shape[-2],size):
  end=min(begin+size,q.shape[-2]);y,s=chunked(q[...,begin:end,:],k[...,begin:end,:],v[...,begin:end,:],g[...,begin:end],b[...,begin:end],s,active[...,begin:end]);ys.append(y)
 return np.concatenate(ys,axis=-2),s

def validate(expected,actual):
 ratio=0.;error=0.
 for e,a in zip(expected,actual):
  assert np.isfinite(a).all()
  diff=np.abs(e-a);error=max(error,float(diff.max()));ratio=max(ratio,float((diff/(np.float32(5e-6)+np.float32(5e-5)*np.abs(e))).max()))
 return ratio,error

if __name__=='__main__':
 output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(852);rows=[]
 for batch in (1,2):
  for n in (2,16,32,40,64,80):
   for mode in ('ordinary','inactive','zero_decay','all_inactive','no_decay','beta_extremes'):
    q=rng.normal(size=(batch,4,n,128)).astype(np.float32);k=rng.normal(size=q.shape).astype(np.float32)
    q/=np.sqrt(np.sum(q*q,axis=-1,keepdims=True,dtype=np.float32)+np.float32(1e-6))*np.sqrt(np.float32(128))
    k/=np.sqrt(np.sum(k*k,axis=-1,keepdims=True,dtype=np.float32)+np.float32(1e-6))
    v=rng.normal(size=(batch,4,n,128)).astype(np.float32);g=-rng.uniform(.001,2,size=(batch,4,n)).astype(np.float32)
    b=rng.uniform(.01,.99,size=g.shape).astype(np.float32);state=rng.normal(0,.1,size=(batch,4,128,128)).astype(np.float32);active=np.ones(g.shape,dtype=bool)
    if mode=='inactive':active[...,::3]=False
    if mode=='all_inactive':active[:]=False
    if mode=='zero_decay':g[...,n//2]=-1000
    if mode=='no_decay':g[:]=0
    if mode=='beta_extremes':b[...,::2]=0;b[...,1::2]=1
    expected=sequential(q,k,v,g,b,state,active)
    for size in (n,32):
     actual=segmented(q,k,v,g,b,state,active,size);ratio,error=validate(expected,actual)
     rows.append(dict(batch=batch,tokens=n,mode=mode,block=size,worst_ratio=ratio,max_abs_error=error,passed=ratio<=1))
     (output/'summary.json').write_text(json.dumps(dict(scope=__doc__,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows,all_passed=all(x['passed'] for x in rows)),indent=2))
     assert ratio<=1,(batch,n,mode,size,ratio)
 print(json.dumps(dict(scope='CPU numerical algebra only; not real weights or GPU speed',cases=len(rows),worst_ratio=max(x['worst_ratio'] for x in rows),all_passed=True)),flush=True)
