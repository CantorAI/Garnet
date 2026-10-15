"""Independent checkpoint-backed Garnet/xModel Metal numerical integration."""
import json,struct,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
rng=np.random.default_rng(351)
dim,intermediate,vocab=12,20,19
weights={
 'model.language_model.norm.weight':rng.normal(size=dim).astype(np.float32),
 'model.language_model.layers.0.mlp.gate_proj.weight':rng.normal(size=(intermediate,dim)).astype(np.float32)*.1,
 'model.language_model.layers.0.mlp.up_proj.weight':rng.normal(size=(intermediate,dim)).astype(np.float32)*.1,
 'model.language_model.layers.0.mlp.down_proj.weight':rng.normal(size=(dim,intermediate)).astype(np.float32)*.1,
 'model.language_model.embed_tokens.weight':rng.normal(size=(vocab,dim)).astype(np.float32)}
header={};parts=[];offset=0
for name,w in weights.items():
 bits=w.view(np.uint32);bf16=((bits+0x7fff+((bits>>16)&1))>>16).astype('<u2')
 weights[name]=(bf16.astype(np.uint32)<<16).view(np.float32)
 data=bf16.tobytes();header[name]=dict(dtype='BF16',shape=list(w.shape),data_offsets=[offset,offset+len(data)])
 parts.append(data);offset+=len(data)
encoded=json.dumps(header,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%8)
with tempfile.TemporaryDirectory(prefix='garnet-metal-graph-') as folder:
 folder=Path(folder);checkpoint=folder/'weights.safetensors'
 checkpoint.write_bytes(struct.pack('<Q',len(encoded))+encoded+b''.join(parts))
 for batch in (1,2):
  x=rng.normal(size=(batch,dim)).astype(np.float32)
  active=np.ones(batch,dtype=np.float32)
  if batch>1:active[-1]=0
  masked=np.where(active[:,None]!=0,x,0)
  n=masked/np.sqrt(np.mean(masked*masked,axis=-1,keepdims=True)+1e-6)*(1+weights['model.language_model.norm.weight'])
  gate=n@weights['model.language_model.layers.0.mlp.gate_proj.weight'].T
  up=n@weights['model.language_model.layers.0.mlp.up_proj.weight'].T
  hidden=(gate/(1+np.exp(-gate))*up)@weights['model.language_model.layers.0.mlp.down_proj.weight'].T
  expected=(masked+hidden)@weights['model.language_model.embed_tokens.weight'].T
  fixture=folder/'fixture.json';fixture.write_text(json.dumps(dict(checkpoint=str(checkpoint),
      shape=[batch,dim],hidden=x.ravel().tolist(),active=active.tolist(),expected=expected.ravel().tolist())))
  subprocess.run([sys.argv[1],str(Path(__file__).with_name('run_model_graph.py')),
      str(fixture),sys.argv[2],sys.argv[3]],check=True)
