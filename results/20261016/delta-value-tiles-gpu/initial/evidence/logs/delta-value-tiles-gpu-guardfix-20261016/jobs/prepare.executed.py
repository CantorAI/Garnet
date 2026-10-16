import sys
sys.path.insert(0,'/Users/zsdx/garnet-metal-investigation/source/test2026/metal')
"""Reproducible primitive inputs, including real Qwen3.5 projections."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from probe_real_chunk_delta_formulation import weights
from reference_decoder import rms_norm
folder=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(814);cases=[]
def save(name,batch,chunk,heads,prepared,state,active,scope):
 case=dict(name=name,batch=batch,chunk=chunk,heads=heads,scope=scope)
 for key,value in [('prepared',prepared),('state',state),('active',active)]:
  p=out/(name+'-'+key+'.bin');value.astype('<f4').tofile(p);case[key]=p.name;case[key+'_sha256']=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
 cases.append(case)
for batch in (1,2):
 for heads in (2,16):
  for chunk in (1,2,16,32,64,80):
   for mode in ('ordinary','inactive','correlated','zero_decay'):
    qw=heads*128;pw=3*qw+2*heads;prepared=rng.normal(0,.3,size=(batch,chunk,pw)).astype(np.float32);state=rng.normal(0,.03,size=(batch,heads,128,128)).astype(np.float32);active=np.ones((batch,chunk),np.float32)
    prepared[...,3*qw:3*qw+heads]=-rng.uniform(.001,2,size=(batch,chunk,heads)).astype(np.float32)
    prepared[...,3*qw+heads:]=rng.uniform(.05,.95,size=(batch,chunk,heads)).astype(np.float32)
    if mode=='inactive':active[:,::3]=0
    if mode=='correlated':prepared[...,qw:2*qw]=prepared[:,0:1,qw:2*qw]
    if mode=='zero_decay':prepared[:,chunk//2,3*qw:3*qw+heads]=-1000
    save(f'synthetic-b{batch}-h{heads}-c{chunk}-{mode}',batch,chunk,heads,prepared,state,active,'synthetic finite raw prepared values')
config=json.loads((folder/'config.json').read_text())['text_config'];kh=config['linear_num_key_heads'];heads=config['linear_num_value_heads'];channels=2*kh*128+heads*128
for layer in (0,config['num_hidden_layers']-2):
 w=weights(folder/'model.safetensors',layer)
 for batch,chunk in ((1,1),(1,80),(2,1),(2,80)):
  qw=heads*128;pw=3*qw+2*heads;prepared=np.empty((batch,chunk,pw),np.float32);conv=np.zeros((batch,channels,config['linear_conv_kernel_dim']),np.float32);active=np.ones((batch,chunk),np.float32);active[:,-1]=0 if chunk>1 else 1
  hidden=rng.normal(size=(batch,chunk,config['hidden_size'])).astype(np.float32)
  for t in range(chunk):
   x=rms_norm(hidden[:,t]*active[:,t,None],w['input_norm']);projected=x@w['qkv'].T;tentative=np.concatenate((conv[:,:,1:],projected[:,:,None]),axis=-1)
   z=np.sum(tentative*w['conv'][:,0,:],axis=-1);mixed=z/(1+np.exp(-z));q,k,v=np.split(mixed,[kh*128,2*kh*128],axis=-1)
   q=np.repeat(q.reshape(batch,kh,128),heads//kh,axis=1);k=np.repeat(k.reshape(batch,kh,128),heads//kh,axis=1)
   prepared[:,t,:qw]=q.reshape(batch,qw);prepared[:,t,qw:2*qw]=k.reshape(batch,qw);prepared[:,t,2*qw:3*qw]=v
   prepared[:,t,3*qw:3*qw+heads]=-np.exp(w['a_log'])*np.logaddexp(0,x@w['a_proj'].T+w['dt_bias'])
   prepared[:,t,3*qw+heads:]=1/(1+np.exp(-(x@w['b_proj'].T)));conv=np.where(active[:,t,None,None]!=0,tentative,conv)
  state=rng.normal(0,.03,size=(batch,heads,128,128)).astype(np.float32)
  save(f'real-layer{layer}-b{batch}-c{chunk}',batch,chunk,heads,prepared,state,active,'actual trained qkv/conv/decay/beta projections; synthetic hidden and prior state')
(out/'manifest.json').write_text(json.dumps(dict(cases=cases,checkpoint_sha256=hashlib.file_digest((folder/'model.safetensors').open('rb'),'sha256').hexdigest(),scope=__doc__),indent=2));print('Prepared cases',len(cases),flush=True)
