"""CPU feasibility of fixed-scale exact half planes; not model/GPU precision admission."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
out=Path(sys.argv[1]).resolve();probe=Path(__file__).with_name('mps_half_triplet_lower_bound.mm')
def examine(bits):
    x=np.asarray(bits,dtype='<u4').view('<f4')
    with np.errstate(over='ignore',invalid='ignore',under='ignore'):
        h0=x.astype('<f2');r=x-h0.astype('<f4');h1=np.ldexp(r,11).astype('<f2');r=r-np.ldexp(h1.astype('<f4'),-11);h2=np.ldexp(r,22).astype('<f2')
        recovered=(h0.astype('<f8')+np.ldexp(h1.astype('<f8'),-11)+np.ldexp(h2.astype('<f8'),-22)).astype('<f4')
    admitted=np.isfinite(x)&np.isfinite(h0)&np.isfinite(h1)&np.isfinite(h2)&(x.view('<u4')==recovered.view('<u4'))
    result=dict(inputs=int(x.size),admitted=int(admitted.sum()),requires_original_f32_fallback=int((~admitted).sum()),
      source_sha256=hashlib.sha256(x.tobytes()).hexdigest(),plane_sha256=[hashlib.sha256(h.tobytes()).hexdigest() for h in (h0,h1,h2)],
      nonzero_plane_values=[int(np.count_nonzero(h)) for h in (h0,h1,h2)],
      scope='Fixed scales 0/11/22, source-bit reconstruction in CPU binary64 then binary32; no GPU split/reconstruction or model activation coverage')
    return result,admitted
profiles=[]
for model,k in [('Qwen3-0.6B',1024),('Qwen2.5-Coder-0.5B-Instruct',896),('Qwen3.5-0.8B',1024)]:
    for rows in (80,160):
        r=np.arange(rows,dtype='<u8')[:,None];j=np.arange(k,dtype='<u8')[None,:]
        base=(np.uint64(0x3f800000)|((j*104729+r*15485863+17)&0x7fffff)).astype('<u4').view('<f4')
        for pattern,scale in enumerate((1,-1,.5,2)):
            x=(base*np.float32(scale)).astype('<f4');record,admitted=examine(x.view('<u4'))
            assert admitted.all(),'Bounded synthetic fixture unexpectedly needs fallback'
            profiles.append(dict(model=model,rows=rows,inner=k,pattern=pattern,scale=scale,**record))
mantissas=(0,1,0x1ff,0x3ff,0x400,0x123456,0x3fffff,0x7ffffe,0x7fffff)
edge_bits=np.asarray([(sign<<31)|(exponent<<23)|mantissa for sign in (0,1) for exponent in range(255) for mantissa in mantissas],dtype='<u4')
edges,admitted=examine(edge_bits)
edge_profiles=[]
for exponent in range(255):
    mask=((edge_bits>>23)&255)==exponent;edge_profiles.append(dict(exponent=exponent,inputs=int(mask.sum()),admitted=int(admitted[mask].sum()),requires_original_f32_fallback=int((~admitted[mask]).sum())))
assert edges['requires_original_f32_fallback']>0
result=dict(helper_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),probe_source_sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),versions=dict(python=sys.version,numpy=np.__version__),profiles=profiles,
 bounded_inputs_checked=sum(p['inputs'] for p in profiles),bounded_source_bits_exact=True,finite_exponent_edge_cases=edges,edge_profiles=edge_profiles,
 precision_policy='Admission requires finite exact original source-bit recovery, otherwise retain original FP32 route. This CPU study does not implement that runtime policy.',
 scope='CPU exact-representation feasibility only. Synthetic bounded inputs and finite exponent/mantissa edges, not universal FP32 coverage or captured model activations; no GPU/arithmetic precision, inference speed or vLLM claim.',full_goal_met=False)
out.write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k not in ('profiles','edge_profiles','versions')}))
