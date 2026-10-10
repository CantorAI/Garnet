import json,subprocess,sys
import numpy as np
from reference_linear import step_linear
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
            out,next_conv,next_state=step_linear(x,conv_state,state,active,w,kh,vh,kd,vd)
            for name,expected in [('out',out),('state',next_state),('conv_state',next_conv)]:
                observed=np.array(actual[name],dtype=np.float32).reshape(expected.shape)
                assert np.isfinite(observed).all()
                np.testing.assert_allclose(observed,expected,rtol=5e-5,atol=5e-6)
            # Feed observed state forward, keeping the independent reference local.
            state=np.array(actual['state'],dtype=np.float32).reshape(state.shape)
            conv_state=np.array(actual['conv_state'],dtype=np.float32).reshape(conv_state.shape)
        print('linear attention projections/convolution/recurrence/gated norm batch',batch,'heads',kh,vh,'PASS')
