"""Real-weight CPU proof for proposed chunk delta math; no GPU speed claim."""
import hashlib,json,mmap,struct,sys
from pathlib import Path
import numpy as np
from probe_chunk_delta_formulation import segmented
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pi5b'))
from reference_decoder import linear_block,rms_norm,finish_block

def weights(model,layer):
 names=dict(input_norm='input_layernorm.weight',post_norm='post_attention_layernorm.weight',mlp_gate='mlp.gate_proj.weight',mlp_up='mlp.up_proj.weight',mlp_down='mlp.down_proj.weight')
 names.update({name:'linear_attn.'+suffix for name,suffix in dict(qkv='in_proj_qkv.weight',z='in_proj_z.weight',b_proj='in_proj_b.weight',a_proj='in_proj_a.weight',conv='conv1d.weight',a_log='A_log',dt_bias='dt_bias',norm='norm.weight',out_proj='out_proj.weight').items()})
 out={}
 with model.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as m:
  size=struct.unpack_from('<Q',m)[0];header=json.loads(m[8:8+size]);base=8+size
  for name,suffix in names.items():
   info=header[f'model.language_model.layers.{layer}.{suffix}'];a,b=info['data_offsets']
   if info['dtype']=='BF16':v=(np.frombuffer(m,dtype='<u2',count=int(np.prod(info['shape'])),offset=base+a).astype(np.uint32)<<16).view(np.float32)
   elif info['dtype']=='F32':v=np.frombuffer(m,dtype='<f4',count=int(np.prod(info['shape'])),offset=base+a).copy()
   else:raise ValueError(info['dtype'])
   out[name]=v.reshape(info['shape'])
 return out

def prepare(hidden,conv,active,w,kh,vh,kd,vd):
 qs=[];ks=[];vs=[];gs=[];bs=[];zs=[];xs=[]
 for t in range(hidden.shape[1]):
  x=hidden[:,t]*active[:,t,None];normalized=rms_norm(x,w['input_norm']);xs.append(x)
  projected=normalized@w['qkv'].T;tentative=np.concatenate((conv[:,:,1:],projected[:,:,None]),axis=-1)
  convolution=np.sum(tentative*w['conv'][:,0,:],axis=-1);mixed=convolution/(1+np.exp(-convolution))
  q,k,v=np.split(mixed,[kh*kd,2*kh*kd],axis=-1)
  q=np.repeat(q.reshape(-1,kh,kd),vh//kh,axis=1);k=np.repeat(k.reshape(-1,kh,kd),vh//kh,axis=1)
  q=q/np.sqrt(np.sum(q*q,axis=-1,keepdims=True)+np.float32(1e-6))/np.sqrt(np.float32(kd))
  k=k/np.sqrt(np.sum(k*k,axis=-1,keepdims=True)+np.float32(1e-6))
  qs.append(q);ks.append(k);vs.append(v.reshape(-1,vh,vd));bs.append(1/(1+np.exp(-(normalized@w['b_proj'].T))))
  gs.append(-np.exp(w['a_log'])*np.logaddexp(0,normalized@w['a_proj'].T+w['dt_bias']))
  zs.append((normalized@w['z'].T).reshape(-1,vh,vd));conv=np.where(active[:,t,None,None],tentative,conv)
 vectors=[np.stack(x,axis=2).astype(np.float32) for x in (qs,ks,vs)]
 scalars=[np.stack(x,axis=2).astype(np.float32) for x in (gs,bs)]
 return vectors+scalars,conv,np.stack(zs,axis=1),np.stack(xs,axis=1)

def ratio(expected,actual):
 assert np.isfinite(actual).all();diff=np.abs(expected-actual)
 return float((diff/(5e-6+5e-5*np.abs(expected))).max()),float(diff.max())

if __name__=='__main__':
 folder=Path(sys.argv[1]);output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=False)
 config=json.loads((folder/'config.json').read_text())['text_config'];model=folder/'model.safetensors';rng=np.random.default_rng(621)
 kh=config['linear_num_key_heads'];vh=config['linear_num_value_heads'];kd=config['linear_key_head_dim'];vd=config['linear_value_head_dim'];channels=2*kh*kd+vh*vd
 rows=[];metadata=dict(scope=__doc__,checkpoint_sha256=hashlib.file_digest(model.open('rb'),'sha256').hexdigest(),source_sha256=hashlib.file_digest(Path(__file__).open('rb'),'sha256').hexdigest(),algebra_source_sha256=hashlib.file_digest(Path(__file__).with_name('probe_chunk_delta_formulation.py').open('rb'),'sha256').hexdigest())
 for layer in (0,config['num_hidden_layers']-2):
  w=weights(model,layer)
  for batch in (1,2):
   for chunk in ((16,32,40,64,80) if batch==1 else (16,32,40,64)):
    conv=np.zeros((batch,channels,config['linear_conv_kernel_dim']),np.float32);state=np.zeros((batch,vh,kd,vd),np.float32)
    for iteration in range(3):
     hidden=rng.normal(size=(batch,chunk,config['hidden_size'])).astype(np.float32);active=np.ones((batch,chunk),dtype=bool)
     if iteration==1:active[:,-1]=False
     if batch==2 and iteration==0:active[1,0]=False
     initial_conv=conv.copy();initial_state=state.copy();expected=[]
     for t in range(chunk):
      y,conv,state=linear_block(hidden[:,t],conv,state,active[:,t],w,kh,vh,kd,vd);state=state.astype(np.float32);conv=conv.astype(np.float32);expected.append(y)
     expected=np.stack(expected,axis=1);inputs,next_conv,z,x=prepare(hidden,initial_conv,active,w,kh,vh,kd,vd)
     for block in (chunk,32):
      core,next_state=segmented(*inputs,initial_state,np.broadcast_to(active[:,None,:],(batch,vh,chunk)),block)
      core=core.transpose(0,2,1,3);normalized=core/np.sqrt(np.mean(core*core,axis=-1,keepdims=True)+1e-6)*w['norm'];gated=normalized*z/(1+np.exp(-z))
      projected=gated.reshape(batch*chunk,vh*vd)@w['out_proj'].T;projected=projected.reshape(batch,chunk,-1);projected=np.where(active[...,None],projected,0)
      actual=finish_block((x+projected).reshape(batch*chunk,-1),w).reshape(batch,chunk,-1)
      hr,he=ratio(expected,actual);sr,se=ratio(state,next_state);cr,ce=ratio(conv,next_conv)
      row=dict(layer=layer,batch=batch,chunk=chunk,iteration=iteration,block=block,hidden_ratio=hr,state_ratio=sr,conv_ratio=cr,max_hidden_error=he,max_state_error=se,passed=max(hr,sr,cr)<=1)
      rows.append(row);(output/'summary.json').write_text(json.dumps(dict(metadata=metadata,rows=rows,all_passed=all(v['passed'] for v in rows)),indent=2));print(json.dumps(row),flush=True)
      assert row['passed'],row
 print('Real-weight CPU formulation proof PASS',len(rows),flush=True)
