"""Derived INT8 screening index; unchanged original BF16 heads and actual native queries."""
import hashlib,json,mmap,os,struct,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]).resolve();model=sys.argv[2];out=Path(sys.argv[3]).resolve();out.mkdir(exist_ok=False)
def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
origin=root/'logs/pruned-head-gpu-20261013/jobs'/(model+'-assets');old=json.loads((origin/'manifest.json').read_text())
assert digest(origin/'raw.bin')==old['raw_sha256'];n,k=old['rows'],old['inner'];assert k in (896,1024)
checkpoint=root/'models'/model/'model.safetensors';config=json.loads((checkpoint.parent/'config.json').read_text())
with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
    size=struct.unpack_from('<Q',mapping)[0];header=json.loads(mapping[8:8+size]);base=8+size
    name='model.language_model.embed_tokens.weight' if 'model.language_model.embed_tokens.weight' in header else 'model.embed_tokens.weight' if config.get('tie_word_embeddings',False) else 'lm_head.weight'
    tensor=header[name];a,b=tensor['data_offsets'];assert tensor['dtype']=='BF16' and tensor['shape']==[n,k] and hashlib.sha256(mapping[base+a:base+b]).hexdigest()==old['raw_sha256']
os.link(origin/'raw.bin',out/'raw.bin');raw=np.memmap(out/'raw.bin',dtype='<u2',mode='r',shape=(n,k))
index=np.memmap(out/'approximate.bin',dtype='i1',mode='w+',shape=(n,k));meta=np.empty((n,4),dtype='<f4');largest=0.0
for start in range(0,n,512):
    values=(np.asarray(raw[start:start+512],np.uint32)<<16).view(np.float32).astype(np.float64);assert np.isfinite(values).all()
    maxima=np.maximum(np.max(np.abs(values),axis=1),np.finfo(np.float32).tiny)
    scale=np.exp2(np.ceil(np.log2(maxima/127)));q=np.rint(values/scale[:,None]);assert np.max(np.abs(q))<=127
    restored=q*scale[:,None];assert np.array_equal(restored,restored.astype(np.float32).astype(np.float64))
    norms=np.stack([np.linalg.norm(values-restored,axis=1),np.linalg.norm(values,axis=1),np.linalg.norm(restored,axis=1)],axis=1)
    upper=np.nextafter((norms*1.0001).astype(np.float32),np.float32(np.inf));assert np.all(upper.astype(np.float64)>=norms)
    index[start:start+len(values)]=q.astype(np.int8);meta[start:start+len(values),:3]=upper;meta[start:start+len(values),3]=scale.astype(np.float32)
    largest=max(largest,float(np.max(norms[:,0]/np.maximum(norms[:,1],1e-30))))
index.flush();meta.tofile(out/'remaining.bin');del index,raw
for query in old['queries']:
    for suffix,field in (('.input.bin','input_sha256'),('.reference.bin','reference_sha256')):
        p=origin/(query['name']+suffix);assert digest(p)==query[field];os.link(p,out/p.name)
manifest=dict(**old);manifest.update(scope=__doc__,checkpoint_sha256=digest(checkpoint),head_tensor=name,approximate_sha256=digest(out/'approximate.bin'),remaining_norm_sha256=digest(out/'remaining.bin'),metadata_layout=['upward residual L2','upward original L2','upward reconstructed L2','exact power-of-two scale'],screen_index='INT8 with per-row dyadic scale; used only for rejection bounds, retained results always original BF16/FP32 dot',largest_relative_weight_L2_error=largest)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(model=model,rows=n,inner=k,queries=len(old['queries']),index_bytes=n*k,largest_relative_weight_L2_error=largest)))
