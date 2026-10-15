"""CPU integer admission diagnostic; no GPU execution or performance claim."""
import hashlib,json,mmap,struct,sys
from pathlib import Path
import numpy as np
checkpoint=Path(sys.argv[1]);output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=False);reports=[]
with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
    size=struct.unpack_from('<Q',mapping)[0];header=json.loads(mapping[8:8+size]);base=8+size
    for name,info in header.items():
        if not name.endswith('linear_attn.in_proj_qkv.weight'):continue
        assert info['dtype']=='BF16';start,end=info['data_offsets']
        bits=np.frombuffer(mapping,dtype='<u2',count=(end-start)//2,offset=base+start).reshape(info['shape']).copy()
        magnitude=bits&0x7fff;largest=magnitude.max(axis=1);row_exp=(largest>>7)&255
        ideal=np.where(largest==0,0,142-row_exp.astype(np.int32))
        row_valid=(largest==0)|((row_exp>0)&(row_exp<255)&(ideal>=-64)&(ideal<=64))
        exponent=((magnitude>>7)&255).astype(np.int32);significand=128+(magnitude&127).astype(np.uint32);policies={}
        for policy,scales in [('current',ideal),('clamped',np.clip(ideal,-64,64))]:
            scaled=exponent-127+scales[:,None]
            good=(magnitude==0)|((exponent>0)&(exponent<255)&(scaled>=-24)&(scaled<=15))
            for shift in range(1,8):good&=~((scaled==-17-shift)&((significand&((1<<shift)-1))!=0))
            if policy=='current':good&=row_valid[:,None]
            rejected=~good;values=(bits[rejected].astype(np.uint32)<<16).view(np.float32)
            policies[policy]=dict(exception_weights=int(rejected.sum()),exception_fraction=float(rejected.mean()),rows_with_exceptions=int(rejected.any(axis=1).sum()),max_exceptions_per_row=int(rejected.sum(axis=1).max()),max_exception_abs=float(np.abs(values).max()) if values.size else 0)
        reports.append(dict(tensor=name,shape=info['shape'],weights=int(bits.size),current_invalid_scale_rows=int((~row_valid).sum()),ideal_scale_min=int(ideal.min()),ideal_scale_max=int(ideal.max()),policies=policies))
assert reports
result=dict(scope=__doc__,checkpoint_sha256=hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest(),source_sha256=hashlib.file_digest(Path(__file__).open('rb'),'sha256').hexdigest(),reports=reports)
(output/'summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(matrices=len(reports),current_exception_weights=sum(v['policies']['current']['exception_weights'] for v in reports),clamped_exception_weights=sum(v['policies']['clamped']['exception_weights'] for v in reports),fully_admitted_after_clamp=sum(v['policies']['clamped']['exception_weights']==0 for v in reports),max_ideal_scale=max(v['ideal_scale_max'] for v in reports))))
