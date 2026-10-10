import json, subprocess, sys
import numpy as np
rng=np.random.default_rng(52)
for batch in (1,2):
    heads,kd,vd=2,7,5
    shape=(batch,heads,kd,vd)
    state=rng.normal(size=shape).astype(np.float32)
    for step in range(5):
        q=rng.normal(size=(batch,heads,kd)).astype(np.float32)
        k=rng.normal(size=q.shape).astype(np.float32)
        v=rng.normal(size=(batch,heads,vd)).astype(np.float32)
        g=-rng.uniform(0,3,size=(batch,heads)).astype(np.float32)
        beta=rng.uniform(0,1,size=g.shape).astype(np.float32)
        active=np.ones(batch,dtype=np.uint8)
        if batch==2 and step>=2: active[1]=0
        if step==4:q[:]=0;k[:]=0
        payload=dict(batch=batch,heads=heads,key_dim=kd,value_dim=vd,active=active.tolist())
        for name,a in [('q',q),('k',k),('v',v),('g',g),('beta',beta),('state',state)]:payload[name]=a.ravel().tolist()
        actual=json.loads(subprocess.check_output([sys.argv[1]],input=json.dumps(payload).encode()))
        q=q/np.sqrt(np.sum(q*q,axis=-1,keepdims=True)+1e-6)/np.sqrt(kd)
        k=k/np.sqrt(np.sum(k*k,axis=-1,keepdims=True)+1e-6)
        next_state=state*np.exp(g)[...,None,None]
        prediction=np.einsum('bhkv,bhk->bhv',next_state,k)
        delta=(v-prediction)*beta[...,None]
        next_state+=k[...,None]*delta[...,None,:]
        out=np.einsum('bhkv,bhk->bhv',next_state,q)
        next_state[active==0]=state[active==0];out[active==0]=0
        np.testing.assert_allclose(np.array(actual['state']).reshape(shape),next_state,rtol=2e-5,atol=2e-6)
        np.testing.assert_allclose(np.array(actual['out']).reshape(out.shape),out,rtol=2e-5,atol=2e-6)
        state=np.array(actual['state'],dtype=np.float32).reshape(shape)
    print(f'batch={batch} recurrent output/state, inactive-row isolation, zero-QK PASS')
