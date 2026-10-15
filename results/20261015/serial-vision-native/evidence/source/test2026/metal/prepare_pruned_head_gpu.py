"""Prepare unchanged BF16 heads and actual native queries for GPU pruning.

Arguments: investigation root, model name, new output directory. No weights
are quantized. Reference logits come from the completed native capture.
"""
import hashlib,json,os,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);model=sys.argv[2];output=Path(sys.argv[3]);output.mkdir(parents=True,exist_ok=False)
if sys.flags.optimize:raise RuntimeError('Numerical validation requires active assertions')

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

origin=root/'logs/lossless-gpu-probe-20261013/jobs'/(model+'-assets')/'matrix-1'
case=json.loads((origin/'case.json').read_text());rows,inner=case['rows'],case['inner'];blocks=inner//128
assert inner%128==0 and digest(origin/'raw.bin')==case['sha256']['raw.bin']
os.link(origin/'raw.bin',output/'raw.bin')
raw=np.memmap(output/'raw.bin',dtype='<u2',mode='r',shape=(rows,inner));remaining=np.empty((rows,blocks+1),np.float32)
for start in range(0,rows,512):
    values=(np.asarray(raw[start:start+512],np.uint32)<<16).view(np.float32).astype(np.float64).reshape(-1,blocks,128)
    assert np.isfinite(values).all()
    squared=np.sum(values*values,axis=2);tail=np.cumsum(squared[:,::-1],axis=1)[:,::-1]
    # Upward FP32 storage and a generous norm allowance. GPU partial/full
    # dot error has an additional separate allowance in the pruning kernel.
    remaining[start:start+len(values),:blocks]=np.nextafter((tail*1.0001).astype(np.float32),np.float32(np.inf))
    remaining[start:start+len(values),blocks]=0
remaining.tofile(output/'remaining.bin');del remaining,raw,values
queries=[];capture_hashes={}
for batch in (1,2):
    source=root/'logs/partial-head-bounds-fusion-20261013/jobs'/(model+'-b'+str(batch))/'queries.json'
    capture_hashes[str(batch)]=digest(source);capture=json.loads(source.read_text());assert capture['width']==inner
    for query in capture['queries']:
        scores=np.asarray(json.loads(Path(query['scores_file']).read_text()),np.float32).reshape(batch,rows)
        hidden=np.asarray(query['normalized'],np.float32).reshape(batch,inner)
        label='b'+str(batch)+'-step'+str(query['step'])
        hidden.tofile(output/(label+'.input.bin'));scores.tofile(output/(label+'.reference.bin'))
        queries.append(dict(name=label,batch=batch,step=query['step'],predictions=query['prediction'],
            input_sha256=digest(output/(label+'.input.bin')),reference_sha256=digest(output/(label+'.reference.bin'))))
    if batch==2:
        # Different actual states in the two rows exercise shared-weight
        # eligibility beyond the duplicated-prompt comparison workload.
        first,last=capture['queries'][0],capture['queries'][-1]
        hidden=np.stack([np.asarray(first['normalized'],np.float32).reshape(batch,inner)[0],np.asarray(last['normalized'],np.float32).reshape(batch,inner)[1]])
        scores=np.stack([np.asarray(json.loads(Path(first['scores_file']).read_text()),np.float32).reshape(batch,rows)[0],np.asarray(json.loads(Path(last['scores_file']).read_text()),np.float32).reshape(batch,rows)[1]])
        label='b2-mixed-first-last';hidden.tofile(output/(label+'.input.bin'));scores.tofile(output/(label+'.reference.bin'))
        queries.append(dict(name=label,batch=2,step=-1,predictions=[first['prediction'][0],last['prediction'][1]],
            input_sha256=digest(output/(label+'.input.bin')),reference_sha256=digest(output/(label+'.reference.bin'))))
manifest=dict(model=model,rows=rows,inner=inner,seed_rows=min(8192,rows),block=128,
    weights='Original exact BF16 bits, hard-linked read-only input to the isolated experiment',
    raw_sha256=digest(output/'raw.bin'),remaining_norm_sha256=digest(output/'remaining.bin'),
    capture_sha256=capture_hashes,queries=queries)
(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k!='queries'}))
