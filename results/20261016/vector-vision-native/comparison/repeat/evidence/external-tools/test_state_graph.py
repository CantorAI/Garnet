"""Independent NumPy state transitions through existing Garnet xModel programs."""
import json,struct,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
rng=np.random.default_rng(352)
with tempfile.TemporaryDirectory(prefix='garnet-metal-state-') as folder:
 folder=Path(folder)
 # The component graphs receive weights/states explicitly, but use the same build API.
 checkpoint=folder/'fixture.safetensors'
 header=json.dumps({'unused':dict(dtype='F32',shape=[1],data_offsets=[0,4])}).encode()
 header+=b' '*((-len(header))%8)
 checkpoint.write_bytes(struct.pack('<Q',len(header))+header+struct.pack('<f',0))
 for batch in (1,2):
  active=np.ones(batch,np.float32)
  if batch==2:active[-1]=0
  def random(shape):return rng.normal(size=shape).astype(np.float32)*.2
  heads,key_dim,value_dim,channels,kernel=2,7,5,9,4
  for kind in ('conv','delta'):
   requests=[]
   state=random((batch,channels,kernel) if kind=='conv' else (batch,heads,key_dim,value_dim))
   for step in range(3):
    if kind=='conv':
     token=random((batch,channels));weight=random((channels,kernel))
     shifted=np.concatenate([state[:,:,1:],token[:,:,None]],axis=-1)
     c=np.sum(shifted*weight[None,:,:],axis=-1)
     output=np.where(active[:,None]!=0,c/(1+np.exp(-c)),0)
     next_state=np.where(active[:,None,None]!=0,shifted,state)
     expected=np.concatenate([output[:,:,None],next_state],axis=-1)
     arrays=[token,weight,state,active]
    else:
     q=random((batch,heads,key_dim));k=random(q.shape);v=random((batch,heads,value_dim))
     # Exercise zero query/key norms on the second step, including active rows.
     if step==1:q.fill(0);k.fill(0)
     g=-np.abs(random((batch,heads)));beta=1/(1+np.exp(-random(g.shape)))
     qn=q/np.sqrt(np.sum(q*q,axis=-1,keepdims=True)+np.float32(1e-6))/np.sqrt(np.float32(key_dim))
     kn=k/np.sqrt(np.sum(k*k,axis=-1,keepdims=True)+np.float32(1e-6))
     decayed=state*np.exp(g)[:,:,None,None]
     memory=np.sum(decayed*kn[:,:,:,None],axis=-2)
     change=(v-memory)*beta[:,:,None]
     updated=decayed+kn[:,:,:,None]*change[:,:,None,:]
     output=np.sum(updated*qn[:,:,:,None],axis=-2)
     output=np.where(active[:,None,None]!=0,output,0)
     next_state=np.where(active[:,None,None,None]!=0,updated,state)
     expected=np.concatenate([output,next_state.reshape(batch,heads,key_dim*value_dim)],axis=-1)
     arrays=[q,k,v,g,beta,state,active]
    shapes=[list(a.shape) for a in arrays]
    requests.append(dict(inputs=[a.ravel().tolist() for a in arrays],packed=expected.ravel().tolist(),
        output=output.ravel().tolist(),state=next_state.ravel().tolist()))
    state=next_state
    # Resume the previously inactive row; then stop the first row.
    if batch==2:active=active[::-1].copy()
   for mode in ('packed','output','state'):
    fixture=folder/'fixture.json';fixture.write_text(json.dumps(dict(kind=kind,mode=mode,shapes=shapes,
        checkpoint=str(checkpoint),requests=requests)))
    subprocess.run([sys.argv[1],str(Path(__file__).with_name('run_state_graph.py')),
        str(fixture),sys.argv[2],sys.argv[3]],check=True)
