"""Exact nibble packing of real decoder projections, without vocabulary heads."""
import hashlib,json,mmap,struct,sys
from pathlib import Path
import numpy as np
from prepare_lossless_gpu_probe import save_case
checkpoint=Path(sys.argv[1]);output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=False);cases=[]
with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
    length=struct.unpack_from('<Q',mapping)[0];header=json.loads(mapping[8:8+length]);base=8+length
    for name,info in header.items():
        if name=='__metadata__' or info['dtype']!='BF16' or len(info['shape'])!=2:continue
        if not name.startswith(('model.layers.0.','model.language_model.layers.0.','model.language_model.layers.3.')):continue
        rows,inner=info['shape'];assert inner%128==0
        bits=np.frombuffer(mapping,dtype='<u2',count=rows*inner,offset=base+info['data_offsets'][0])
        case=save_case(output,'matrix-'+str(len(cases)),bits,rows,inner);del bits
        case['tensor']=name;(output/case['name']/'case.json').write_text(json.dumps(case,indent=2)+'\n');cases.append(case)
assert cases
with checkpoint.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
(output/'manifest.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),checkpoint_sha256=sha,
    scope='Actual decoder projection matrices; same BF16 values, no model/head replacement',cases=cases),indent=2)+'\n')
