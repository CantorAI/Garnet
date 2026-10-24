"""Independent bounded-memory CPU decoder-layer oracle for pinned INT4 G128."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import prepare_real_layer_probe as reference
from prepare_real_mlp_probe import bf16

def project(cp,prefix,x):
 if prefix+'.weight_packed' not in cp.index:return ORIGINAL(cp,prefix,x)
 raw,e=cp.tensor(prefix+'.weight_packed');shape,se=cp.tensor(prefix+'.weight_shape');n,k=map(int,shape.view('<i8'))
 if e['dtype']!='I32' or e['shape']!=[n,k//8] or se['dtype']!='I64' or k%128 or len(x)!=k:raise ValueError('Wrong actual packed projection')
 scales,entry=cp.tensor(prefix+'.weight_scale')
 if entry['dtype']!='BF16' or entry['shape']!=[n,k//128]:raise ValueError('Wrong actual scale projection')
 scales=(scales.view('<u2').astype(np.uint32)<<16).view(np.float32).reshape(n,k//128)
 if not np.all(np.isfinite(scales)) or np.any(scales<=0):raise ValueError('Invalid INT4 scales')
 raw=raw.reshape(n,k//2);activation=bf16(x);out=np.empty(n,dtype=np.float32)
 for start in range(0,n,128):
  stop=min(n,start+128);codes=np.empty((stop-start,k),dtype=np.int16)
  codes[:,0::2]=raw[start:stop]&15;codes[:,1::2]=raw[start:stop]>>4;codes-=8
  weight=codes.astype(np.float32)*np.repeat(scales[start:stop],128,axis=1)
  out[start:stop]=weight@activation
 return bf16(out)
ORIGINAL=reference.project
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 if a.output.exists():raise ValueError('Preserve actual layer oracle')
 reference.project=project
 result=reference.prepare(a.checkpoint,capacity=4096)
 result['expected_weight_precision']='ct_int4_g128';result['checkpoint_revision']='4e2c2d8f592d5641beefa6a75357f43cf3756392'
 result['scope']='Pinned INT4 decoder layers0/3, two stateful serial tokens each; all hidden values and sampled states only. Independent block-dequantized CPU arithmetic. Not full-model quality or performance.'
 a.output.write_text(json.dumps(result));print(json.dumps({'oracle_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'scope':result['scope']}),flush=True)
