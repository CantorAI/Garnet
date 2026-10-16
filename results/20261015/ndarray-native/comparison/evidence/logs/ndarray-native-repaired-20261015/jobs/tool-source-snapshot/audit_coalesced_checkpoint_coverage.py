"""CPU-only exact byte/nibble reconstruction across all eligible decoder checkpoint matrices."""
import hashlib,json,mmap,struct,sys,time
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=False)
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
report=dict(scope='CPU checkpoint coverage only. Decoder-layer name/shape admission, one representation per tensor; graph usage/sharing, actual GPU allocations, RSS and performance are not inferred.',source_sha256=digest(Path(__file__)),models=[],full_goal_met=False)
(out/'source.py').write_bytes(Path(__file__).read_bytes());started=time.monotonic()
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    checkpoint=root/'models'/model/'model.safetensors';checkpoint_sha=digest(checkpoint);cases=[]
    with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
        size=struct.unpack_from('<Q',mapping)[0];header=json.loads(mapping[8:8+size]);base=8+size
        for name,tensor in header.items():
            if name=='__metadata__' or not name.startswith(('model.layers.','model.language_model.layers.')) or tensor['dtype']!='BF16' or len(tensor['shape'])!=2:continue
            n,k=tensor['shape'];special=name.startswith('model.layers.') and name.endswith('.self_attn.k_proj.weight')
            if special or not(256<n<=8192) or k%128 or n*k<1048576:continue
            lo,hi=tensor['data_offsets'];raw=mapping[base+lo:base+hi];assert len(raw)==n*k*2
            bits=np.frombuffer(raw,dtype='<u2').reshape(n,k);assert not np.any(((bits>>7)&255)==255)
            padded=(n+3)//4*4
            if padded!=n:bits=np.pad(bits,((0,padded-n),(0,0)))
            blocks=bits.reshape(padded//4,4,k//128,128).transpose(0,2,3,1).reshape(-1,512)
            exponent=(blocks>>7)&255;lower=exponent.min(axis=1);upper=exponent.max(axis=1);compact=(upper-lower)<=15;selected=blocks[compact]
            sm=((selected&127)|((selected>>8)&128)).astype(np.uint8);tags=(((selected>>7)&255)-lower[compact,None]).astype(np.uint8)
            assert np.all(tags<=15);packed=(tags[:,0::2]|(tags[:,1::2]<<4)).astype(np.uint8)
            restored=np.empty(selected.shape,dtype=np.uint16);restored[:,0::2]=packed&15;restored[:,1::2]=packed>>4
            recovered=(sm.astype(np.uint16)&127)|((sm.astype(np.uint16)&128)<<8)|((restored+lower[compact,None])<<7);assert np.array_equal(recovered,selected)
            count=len(blocks);fast=int(compact.sum());extra=count*772 if fast else 0
            cases.append(dict(tensor=name,rows=n,inner=k,raw_sha256=hashlib.sha256(raw).hexdigest(),blocks=count,compact_blocks=fast,exact_byte_plane_roundtrip=True,original_bytes=n*k*2,native_interleaved_bytes=padded*k*2,additional_compact_bytes_one_copy=extra))
    total=sum(c['blocks'] for c in cases);fast=sum(c['compact_blocks'] for c in cases)
    record=dict(model=model,checkpoint_sha256=checkpoint_sha,cases=cases,eligible_matrices=len(cases),blocks=total,compact_blocks=fast,compact_fraction=fast/total,additional_compact_bytes_one_copy=sum(c['additional_compact_bytes_one_copy'] for c in cases),original_eligible_bytes=sum(c['original_bytes'] for c in cases))
    report['models'].append(record);report['elapsed_seconds']=time.monotonic()-started;(out/'report.json').write_bytes((json.dumps(report,indent=2)+'\n').encode());print(model,record['eligible_matrices'],'matrices; compact',round(record['compact_fraction'],6),'additional one-copy MiB',round(record['additional_compact_bytes_one_copy']/2**20,2),flush=True)
