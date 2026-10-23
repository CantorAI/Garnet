"""Independent bounded-memory NumPy oracle for a real packed checkpoint MLP."""
import argparse,hashlib,json,struct
from pathlib import Path
import numpy as np

def bf16(x):
    x=np.asarray(x,dtype=np.float32);bits=x.view(np.uint32)
    return ((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)

def fp8_table():
    c=np.arange(127,dtype=np.int32);e=c>>3;m=c&7
    return np.where(e==0,np.ldexp(m.astype(np.float32),-9),np.ldexp(1+m.astype(np.float32)/8,e-7)).astype(np.float32)

def nearest(values,table):
    high=np.minimum(np.searchsorted(table,values),len(table)-1);low=np.maximum(high-1,0)
    dh=np.abs(table[high]-values);dl=np.abs(values-table[low])
    return np.where((dh<dl)|((dh==dl)&((high&1)==0)),high,low)

class Checkpoint:
    def __init__(self,path):
        self.path=path;self.index=json.loads((path/'model.safetensors.index.json').read_text())['weight_map'];self.headers={}
    def tensor(self,name):
        path=self.path/self.index[name]
        if path.name not in self.headers:
            with path.open('rb') as f:
                n=struct.unpack('<Q',f.read(8))[0]
                if n>16*1024*1024:raise ValueError('Oversized header')
                self.headers[path.name]=(8+n,json.loads(f.read(n)))
        base,header=self.headers[path.name];entry=header[name];begin,end=entry['data_offsets']
        raw=np.memmap(path,mode='r',dtype=np.uint8,offset=base+begin,shape=(end-begin,))
        return raw,entry
    def scalar(self,name):
        raw,e=self.tensor(name)
        if e['dtype']!='F32' or e['shape']!=[] or len(raw)!=4:raise ValueError(name)
        value=float(raw.view('<f4')[0])
        if not np.isfinite(value) or value<=0:raise ValueError(name)
        return value

def project(checkpoint,prefix,x):
    raw,e=checkpoint.tensor(prefix+'.weight');n,packed=e['shape'];k=packed*2
    if e['dtype']!='U8' or len(x)!=k:raise ValueError('Expected packed NVFP4 projection')
    scale,se=checkpoint.tensor(prefix+'.weight_scale')
    if se['shape']!=[n,k//16] or se['dtype'] not in ('F8_E4M3','F8_E4M3FN'):raise ValueError('Invalid scale layout')
    global_w=checkpoint.scalar(prefix+'.weight_scale_2');global_x=checkpoint.scalar(prefix+'.input_scale')
    table=fp8_table();levels=np.array([0,.5,1,1.5,2,3,4,6],dtype=np.float32)
    blocks=bf16(x).reshape(-1,16);maximum=np.max(np.abs(blocks),axis=1)
    sx=table[nearest(maximum/(6*global_x),table)]*global_x
    divided=np.divide(blocks,sx[:,None],out=np.zeros_like(blocks),where=sx[:,None]!=0)
    q=levels[nearest(np.abs(divided),levels)]*np.where(np.signbit(divided),-1,1)
    activation=(q*sx[:,None]).reshape(k).astype(np.float32)
    out=np.empty(n,dtype=np.float32)
    raw=raw.reshape(n,packed);scale=scale.reshape(n,k//16)
    for start in range(0,n,128):
        stop=min(start+128,n);codes=np.empty((stop-start,k),dtype=np.uint8)
        codes[:,0::2]=raw[start:stop]&15;codes[:,1::2]=raw[start:stop]>>4
        sc=scale[start:stop]
        if np.any(sc>126):raise ValueError('Invalid block scale')
        weight=levels[codes&7]*np.where(codes&8,-1,1)
        weight=weight.astype(np.float32)*np.repeat(table[sc]*global_w,16,axis=1)
        out[start:stop]=weight@activation
    return bf16(out)

def prepare(path,layer=0):
    checkpoint=Checkpoint(path);prefix='model.language_model.layers.'+str(layer)+'.mlp'
    _,e=checkpoint.tensor(prefix+'.gate_proj.weight');width=e['shape'][1]*2
    x=np.array([((i*17)%101-50)/50 for i in range(width)],dtype=np.float32)
    gate=project(checkpoint,prefix+'.gate_proj',x);up=project(checkpoint,prefix+'.up_proj',x)
    mid=(gate/(1+np.exp(-gate)))*up
    expected=project(checkpoint,prefix+'.down_proj',mid)
    return {'layer':layer,'input':x.tolist(),'expected':expected.tolist(),'numpy_version':np.__version__,
        'scope':'Real checkpoint MLP only, independent CPU packed-weight/activation quantization; no full-model or performance claim'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path);p.add_argument('--layer',type=int,default=0);a=p.parse_args()
    result=prepare(a.checkpoint,a.layer);a.output.write_text(json.dumps(result))
    print(json.dumps({'oracle_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'outputs':len(result['expected']),'scope':result['scope']}))
