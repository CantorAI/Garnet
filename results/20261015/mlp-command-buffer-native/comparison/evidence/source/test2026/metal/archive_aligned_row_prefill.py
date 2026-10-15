"""Freeze aligned-row results and independently derive common-half admission."""
from pathlib import Path
wrapper=Path(__file__).with_name('archive_interleaved_bf16_body.py').read_text()
ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns)
source=ns['source'].replace("['interleaved-bf16-body-20261015']","['aligned-row-prefill-20261015']")
needle='proof=dict(files={}';assert source.count(needle)==1
source=source.replace(needle,'''import numpy as np
admission={}
for trial in jobs:
    meta=json.loads((root/'logs'/trial/'jobs/metadata.json').read_text());admission[trial]={}
    for model,assets in meta['assets'].items():
        cases=[]
        for case in assets['cases']:
            path=root/'logs/decoder-codec-gpu-20261013/jobs'/(model+'-assets')/case['name']/'raw.bin'
            assert digest_checkpoint(path)==case['sha256']['raw.bin']
            bits=np.fromfile(path,dtype='<u2').astype(np.int32);mag=bits&32767;largest=int(mag.max());e=(largest>>7)&255
            scale=0 if largest==0 else 142-e
            normal=(mag>>7)&255;exponent=normal-127+scale;nonzero=mag!=0
            valid=(~nonzero)|((normal!=0)&(normal!=255)&(exponent>=-24)&(exponent<=15))
            low=nonzero&(exponent<-17);discarded=np.clip(-exponent-17,0,24)
            valid&=(~low)|(((128+(mag&127))&((1<<discarded)-1))==0)
            special=case['tensor'].startswith('model.layers.') and case['tensor'].endswith('.self_attn.k_proj.weight')
            admitted=not special and bits.nbytes//2<=16*1024*1024 and (largest==0 or e not in (0,255)) and -64<=scale<=64 and bool(valid.all())
            cases.append(dict(name=case['name'],global_half_ready=admitted,common_scale=scale,raw_sha256=case['sha256']['raw.bin']))
        admission[trial][model]=cases
'''+needle)
source=source.replace('proof=dict(files={},','proof=dict(aligned_global_half_cases=admission,files={},')
exec(compile(source,__file__,'exec'))
