"""CPU-only exact admission for four-output, 128-K BF16 block storage."""
import hashlib,json,mmap,struct,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=False)
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
report=dict(scope='CPU admission only, no GPU speed or inference claim. Four output channels and128 K terms share a block; compact payload512 sign/mantissa bytes plus256 exponent-tag bytes, original raw four-output fallback.',source_sha256=digest(Path(__file__)),models=[],full_goal_met=False)
(out/'source.py').write_bytes(Path(__file__).read_bytes())
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    folder=root/'logs/decoder-codec-gpu-20261013/jobs'/(model+'-assets');assets=json.loads((folder/'manifest.json').read_text());checkpoint=root/'models'/model/'model.safetensors';assert digest(checkpoint)==assets['checkpoint_sha256'];cases=[]
    with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
        size=struct.unpack_from('<Q',mapping)[0];header=json.loads(mapping[8:8+size]);base=8+size
        for case in assets['cases']:
            path=folder/case['name']/'raw.bin';assert digest(path)==case['sha256']['raw.bin'];tensor=header[case['tensor']];lo,hi=tensor['data_offsets'];assert tensor['dtype']=='BF16' and tensor['shape']==[case['rows'],case['inner']] and hashlib.sha256(mapping[base+lo:base+hi]).hexdigest()==case['sha256']['raw.bin']
            n,k=case['rows'],case['inner'];special=case['tensor'].startswith('model.layers.') and case['tensor'].endswith('.self_attn.k_proj.weight')
            if special or n<=256 or n>8192 or n%4 or k%128 or n*k<1048576:continue
            bits=np.fromfile(path,dtype='<u2').reshape(n,k);blocks=bits.reshape(n//4,4,k//128,128).transpose(0,2,3,1);exponents=(blocks>>7)&255;lower=exponents.min(axis=(2,3));upper=exponents.max(axis=(2,3));compact=(upper-lower)<=15
            sm=((blocks&127)|((blocks>>8)&128)).astype(np.uint16);tags=exponents-lower[:,:,None,None];recovered=(sm&127)|((sm&128)<<8)|((lower[:,:,None,None]+tags)<<7)
            assert np.array_equal(recovered[compact],blocks[compact]);count=compact.size;fast=int(compact.sum());raw=n*k*2
            cases.append(dict(name=case['name'],tensor=case['tensor'],rows=n,inner=k,raw_sha256=case['sha256']['raw.bin'],blocks=count,compact_blocks=fast,compact_fraction=fast/count,cpu_original_bits_exact=True,raw_bytes=raw,fixed_payload_bytes=count*768+count*4,hybrid_payload_bytes=fast*768+(count-fast)*1024+count*8))
    report['models'].append(dict(model=model,checkpoint_sha256=assets['checkpoint_sha256'],cases=cases));print(model,'eligible',len(cases),'compact fraction',sum(c['compact_blocks'] for c in cases)/sum(c['blocks'] for c in cases),flush=True)
    (out/'report.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
