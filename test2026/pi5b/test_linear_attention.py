import json,subprocess,sys
import numpy as np
rng=np.random.default_rng(530)
softplus_x=np.concatenate((np.linspace(-100,100,2001),rng.uniform(-20,20,1024),[-1000,1000,-1e-8,0,1e-8])).astype(np.float32)
softplus=json.loads(subprocess.check_output([sys.argv[1]],input=json.dumps(dict(kind='softplus',x=softplus_x.tolist())).encode()))
np.testing.assert_allclose(np.array(softplus['out']),np.logaddexp(0,softplus_x),rtol=1e-6,atol=2e-6)
print('ARM FP32 softplus range/extreme regression PASS')
def silu(x):return x/(1+np.exp(-x))
for batch in (1,2):
    for kh,vh in ((2,2),(1,2)):
        kd,vd,d,ks=7,5,11,4
        channels=2*kh*kd+vh*vd
        shapes=dict(qkv=(channels,d),z=(vh*vd,d),b_proj=(vh,d),a_proj=(vh,d),conv=(channels,1,ks),a_log=(vh,),dt_bias=(vh,),norm=(vd,),out_proj=(d,vh*vd))
        w={name:rng.normal(0,.2,size=s).astype(np.float32) for name,s in shapes.items()}
        w['norm']+=1
        state=rng.normal(0,.2,size=(batch,vh,kd,vd)).astype(np.float32)
        conv_state=rng.normal(0,.2,size=(batch,channels,ks)).astype(np.float32)
        for step in range(7):
            x=rng.normal(size=(batch,d)).astype(np.float32)
            active=np.ones(batch,dtype=np.float32)
            if batch==2 and 2<=step<=4:active[1]=0
            payload=dict(batch=batch,key_heads=kh,value_heads=vh,key_dim=kd,value_dim=vd,hidden_dim=d,kernel_size=ks)
            for name,a in dict(w,x=x,state=state,conv_state=conv_state,active=active).items():payload[name]=a.ravel().tolist()
            actual=json.loads(subprocess.check_output([sys.argv[1]],input=json.dumps(payload).encode()))
            projected=x@w['qkv'].T
            next_conv=np.concatenate((conv_state[:,:,1:],projected[:,:,None]),axis=-1)
            mixed=silu(np.sum(next_conv*w['conv'][:,0,:],axis=-1))
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
            gated=normalized*silu((x@w['z'].T).reshape(batch,vh,vd))
            out=gated.reshape(batch,vh*vd)@w['out_proj'].T
            next_state[active==0]=state[active==0];next_conv[active==0]=conv_state[active==0];out[active==0]=0
            for name,expected in [('out',out),('state',next_state),('conv_state',next_conv)]:
                observed=np.array(actual[name],dtype=np.float32).reshape(expected.shape)
                assert np.isfinite(observed).all()
                np.testing.assert_allclose(observed,expected,rtol=5e-5,atol=5e-6)
            # Feed observed state forward, keeping the independent reference local.
            state=np.array(actual['state'],dtype=np.float32).reshape(state.shape)
            conv_state=np.array(actual['conv_state'],dtype=np.float32).reshape(conv_state.shape)
        print('linear attention projections/convolution/recurrence/gated norm batch',batch,'heads',kh,vh,'PASS')
