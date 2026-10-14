"""Actual first-layer and full-head BF16 matrices for isolated decode tile tests."""
import hashlib,json,mmap,struct,sys
from pathlib import Path
import numpy as np
checkpoint=Path(sys.argv[1]);output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=False)
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

cases=[]
with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
    length=struct.unpack_from('<Q',mapping)[0];header=json.loads(mapping[8:8+length]);base=8+length
    config=json.loads((checkpoint.parent/'config.json').read_text())
    head='model.language_model.embed_tokens.weight' if 'model.language_model.embed_tokens.weight' in header else 'model.embed_tokens.weight' if config.get('tie_word_embeddings',False) else 'lm_head.weight'
    for tensor,info in header.items():
        if tensor=='__metadata__' or info['dtype']!='BF16' or len(info['shape'])!=2:continue
        if not tensor.startswith(('model.layers.0.','model.language_model.layers.0.')) and tensor!=head:continue
        rows,inner=info['shape'];label='matrix-'+str(len(cases));target=output/(label+'.bin')
        start,end=info['data_offsets'];target.write_bytes(mapping[base+start:base+end])
        cases.append(dict(name=label,tensor=tensor,columns=rows,inner=inner,sha256=digest(target)))
assert cases and any(case['tensor']==head for case in cases), 'Runtime head was not collected as BF16'
# Irregular shapes exercise zero-padded tiles and different output strides.
for inner,columns in ((37,19),(129,65),(132,37)):
    rng=np.random.default_rng(inner);values=rng.normal(size=(columns,inner)).astype(np.float32)
    bits=(values.view(np.uint32)>>16).astype('<u2');label='synthetic-'+str(inner)
    target=output/(label+'.bin');bits.tofile(target)
    cases.append(dict(name=label,tensor='synthetic finite BF16 tail fixture',columns=columns,inner=inner,sha256=digest(target)))
manifest=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
    scope='Actual BF16 matrices, FP32 inputs/accumulation; isolated decode only',cases=cases)
(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(matrices=len(cases),checkpoint_sha256=manifest['checkpoint_sha256'])))
