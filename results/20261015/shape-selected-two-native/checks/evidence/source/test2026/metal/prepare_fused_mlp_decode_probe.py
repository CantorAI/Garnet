"""First-layer real gate/up pairs and irregular independent BF16 fixtures."""
import hashlib,json,mmap,struct,sys
from pathlib import Path
import numpy as np
checkpoint=Path(sys.argv[1]);output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=False);cases=[]
with checkpoint.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapping:
 size=struct.unpack_from('<Q',mapping)[0];header=json.loads(mapping[8:8+size]);base=8+size
 name=next(name for name in header if name.endswith('layers.0.mlp.gate_proj.weight'))
 up=name.replace('gate_proj.weight','up_proj.weight');gate=header[name];other=header[up]
 assert gate['dtype']==other['dtype']=='BF16' and gate['shape']==other['shape']
 for label,info in [('real-gate',gate),('real-up',other)]:
  a,b=info['data_offsets'];(output/(label+'.bin')).write_bytes(mapping[base+a:base+b])
 columns,inner=gate['shape'];cases.append(dict(name='real-mlp',columns=columns,inner=inner,gate_file='real-gate.bin',up_file='real-up.bin',gate_tensor=name,up_tensor=up))
for inner,columns in ((37,19),(129,65),(132,37)):
 rng=np.random.default_rng(inner);name='synthetic-'+str(inner)
 for kind in ('gate','up'):
  values=rng.normal(size=(columns,inner)).astype(np.float32);bits=(values.view(np.uint32)>>16).astype('<u2');bits.tofile(output/(name+'-'+kind+'.bin'))
 cases.append(dict(name=name,columns=columns,inner=inner,gate_file=name+'-gate.bin',up_file=name+'-up.bin',gate_tensor='synthetic',up_tensor='synthetic'))
for c in cases:
 for kind in ('gate','up'):
  p=output/c[kind+'_file'];c[kind+'_sha256']=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
manifest=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest(),scope='MLP pair component only',cases=cases)
(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(cases=len(cases))))
