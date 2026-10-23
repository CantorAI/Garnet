"""Read bounded actual checkpoint slices for an independent sparse head oracle."""
import argparse, hashlib, json, math, struct
from pathlib import Path


def prepare(root,native_fp4=False):
    index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    headers={}
    def read(name, offset=0, size=None):
        path=root/index[name]
        if path.name not in headers:
            with path.open('rb') as f:
                length=struct.unpack('<Q',f.read(8))[0]
                if length>16*1024*1024: raise ValueError('Oversized header')
                headers[path.name]=(8+length,json.loads(f.read(length)))
        start,header=headers[path.name]
        record=header[name];begin,end=record['data_offsets']
        size=end-begin if size is None else size
        if offset<0 or offset+size>end-begin: raise ValueError('Slice exceeds tensor')
        with path.open('rb') as f: f.seek(start+begin+offset);raw=f.read(size)
        if len(raw)!=size: raise ValueError('Truncated tensor slice')
        return raw,record
    raw,record=read('lm_head.weight',0,1)
    rows,packed_width=record['shape'];width=packed_width*2
    if record['dtype']!='U8' or width%16: raise ValueError('Unexpected packed head')
    raw,record=read('lm_head.weight_scale_2')
    if record['dtype']!='F32' or record['shape']!=[]: raise ValueError('Unexpected scale2')
    scale2=struct.unpack('<f',raw)[0]
    def fp8(code):
        e=(code>>3)&15;m=code&7
        if code>126: raise ValueError('Invalid block scale')
        return math.ldexp(1+m/8,e-7) if e else math.ldexp(m,-9)
    positions=[0,1,16,width-1];values=[1.,2.,-3.,.5]
    quantized_values=values
    if native_fp4:
        import numpy as np
        from prepare_real_mlp_probe import fp8_table,nearest,bf16
        raw,entry=read('lm_head.input_scale');global_x=struct.unpack('<f',raw)[0]
        if entry['dtype']!='F32' or entry['shape']!=[] or not math.isfinite(global_x) or global_x<=0:raise ValueError('Invalid activation scale')
        x=np.zeros(width,dtype=np.float32);x[positions]=values;blocks=bf16(x).reshape(-1,16)
        table=fp8_table();levels=np.array([0,.5,1,1.5,2,3,4,6],dtype=np.float32)
        sx=table[nearest(np.max(np.abs(blocks),axis=1)/(6*global_x),table)]*global_x
        divided=np.divide(blocks,sx[:,None],out=np.zeros_like(blocks),where=sx[:,None]!=0)
        q=levels[nearest(np.abs(divided),levels)]*np.where(np.signbit(divided),-1,1)
        quantized_values=(q*sx[:,None]).reshape(-1)[positions].tolist()
    samples=[]
    for row in [0,1,2,17,42,100,1024,8191,rows//2,rows-2,rows-1]:
        expected=0.;payload=[]
        for col,value in zip(positions,quantized_values):
            packed,_=read('lm_head.weight',row*packed_width+col//2,1)
            scale,_=read('lm_head.weight_scale',row*(width//16)+col//16,1)
            code=(packed[0]>>(4*(col%2)))&15
            weight=[0,.5,1,1.5,2,3,4,6][code&7]*(-1 if code&8 else 1)
            expected+=value*weight*fp8(scale[0])*scale2
            payload.append({'column':col,'packed_hex':packed.hex(),'scale_hex':scale.hex()})
        samples.append({'row':row,'expected_f64':expected,'source_bytes':payload})
    return {'shape':[rows,width],'positions':positions,'values':values,'quantized_values':quantized_values,'scale2':scale2,
            'head_mode':'nvfp4_w4a4' if native_fp4 else 'nvfp4_w4a16',
            'samples':samples,'scope':'Actual checkpoint sparse output-head arithmetic only; no full-model or performance claim'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path);p.add_argument('--native-fp4',action='store_true')
    a=p.parse_args();result=prepare(a.checkpoint,a.native_fp4)
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({'oracle_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'samples':len(result['samples'])}))
