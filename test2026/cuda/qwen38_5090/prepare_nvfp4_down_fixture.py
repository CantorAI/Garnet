"""Bounded actual pinned weights plus synthetic signed packed activations; no CUDA."""
import argparse,hashlib,json,math,struct
from pathlib import Path
import numpy as np

PREFIX='model.language_model.layers.0.mlp.down_proj'
REVISION='482ca0f3832238542f8f5295dde86b5f22711d80'

def tiled(scales):
    rows,blocks=scales.shape;padded_rows=(rows+127)//128*128;padded_blocks=(blocks+3)//4*4
    r=np.arange(rows,dtype=np.int64)[:,None];b=np.arange(blocks,dtype=np.int64)[None,:]
    offsets=(r//128)*128*padded_blocks+(b//4)*512+(r%32)*16+((r%128)//32)*4+b%4
    result=np.zeros(padded_rows*padded_blocks,dtype=np.uint8);result[offsets]=scales
    return result.tobytes()

def fp4(raw,rows,k):
    packed=np.frombuffer(raw,dtype=np.uint8).reshape(rows,k//2)
    codes=np.empty((rows,k),dtype=np.uint8);codes[:,0::2]=packed&15;codes[:,1::2]=packed>>4
    lut=np.array([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],dtype=np.float64)
    return lut[codes]

def e4m3(raw):
    codes=np.asarray(raw,dtype=np.uint8);assert np.all(codes<=126)
    exponent=(codes>>3).astype(np.int32);mantissa=(codes&7).astype(np.float64)
    return np.where(exponent==0,mantissa*2.**-9,(1+mantissa/8)*np.exp2(exponent-7))

def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    index_raw=(a.checkpoint/'model.safetensors.index.json').read_bytes();index=json.loads(index_raw)['weight_map'];sources={}
    def tensor(suffix,dtype,shape):
        name=PREFIX+suffix;shard=index[name];assert Path(shard).name==shard
        with (a.checkpoint/shard).open('rb') as f:
            n=struct.unpack('<Q',f.read(8))[0];assert 0<n<=64<<20
            header_raw=f.read(n);header=json.loads(header_raw);entry=header[name]
            assert entry['dtype'] in dtype and entry['shape']==shape
            left,right=entry['data_offsets'];assert 0<=left<right and 8+n+right<=(a.checkpoint/shard).stat().st_size
            f.seek(8+n+left);raw=f.read(right-left);assert len(raw)==right-left
        sources[name]={'shard':shard,'header_sha256':hashlib.sha256(header_raw).hexdigest(),'data_offsets':[left,right],'dtype':entry['dtype'],'shape':shape,'sha256':hashlib.sha256(raw).hexdigest()}
        return raw
    n,k,batch=5120,17408,128
    w=tensor('.weight',{'U8'},[n,k//2]);ws=tensor('.weight_scale',{'F8_E4M3','F8_E4M3FN'},[n,k//16])
    weight_scale=struct.unpack('<f',tensor('.weight_scale_2',{'F32'},[]))[0]
    input_scale=struct.unpack('<f',tensor('.input_scale',{'F32'},[]))[0]
    assert all(math.isfinite(v) and v>0 for v in (weight_scale,input_scale))
    alpha=float(np.float32(weight_scale)*np.float32(input_scale));assert math.isfinite(alpha) and alpha>0
    assert len(w)==n*k//2 and len(ws)==n*k//16
    scale_matrix=np.frombuffer(ws,dtype=np.uint8).reshape(n,k//16);assert np.all(scale_matrix<=126)
    rng=np.random.Generator(np.random.PCG64(20261024));x=rng.integers(0,256,size=(batch,k//2),dtype=np.uint8)
    xs=np.array([56,64,72],dtype=np.uint8)[rng.integers(0,3,size=(batch,k//16))]
    selected=np.linspace(0,n-1,32,dtype=np.int64)
    packed=np.frombuffer(w,dtype=np.uint8).reshape(n,k//2)[selected].tobytes()
    dw=fp4(packed,len(selected),k)*np.repeat(e4m3(scale_matrix[selected]),16,axis=1)
    dx=fp4(x.tobytes(),batch,k)*np.repeat(e4m3(xs),16,axis=1)
    expected=(dx@dw.T)*alpha
    assert np.isfinite(expected).all() and (dw<0).any() and (dx<0).any()
    files={'weight.bin':w,'weight-scales-row.bin':ws,'weight-scales-tiled.bin':tiled(scale_matrix),'input.bin':x.tobytes(),'input-scales-row.bin':xs.tobytes(),'input-scales-tiled.bin':tiled(xs),'expected.f64':expected.astype('<f8').tobytes()}
    for name,raw in files.items():(a.output/name).write_bytes(raw)
    report={'schema':1,'checkpoint_revision':REVISION,'prefix':PREFIX,'n':n,'k':k,'max_batch':batch,'alpha':alpha,'input_scale':input_scale,'weight_scale_2':weight_scale,'selected_rows':selected.tolist(),'source_index_sha256':hashlib.sha256(index_raw).hexdigest(),'source_tensors':sources,'numpy_version':np.__version__,'input_generation':'PCG64 seed20261024; arbitrary signed E2M1 packed codes and scales1/2/4; not recorded model activations','oracle':'float64 decoded actual weights/scales times signed synthetic packed inputs/scales, global alpha rounded once to float32 as production','tolerance':{'absolute':1e-5,'relative':.0078125},'files':{name:hashlib.sha256(raw).hexdigest() for name,raw in files.items()},'scope':'One full pinned down-projection; CPU oracle32 selected output channels x128 rows. Not model-token quality or actual activation distribution.'}
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps({'status':'fixture_prepared','files':len(files),'oracle_outputs':expected.size,'alpha':alpha}))

if __name__=='__main__':main()
