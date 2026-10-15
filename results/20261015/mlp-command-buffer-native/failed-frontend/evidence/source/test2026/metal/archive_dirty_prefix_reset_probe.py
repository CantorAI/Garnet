"""Archive a terminal reset trial, exact sources and omitted binary identity."""
import hashlib,io,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();folder=Path(sys.argv[2]).resolve();target=Path(sys.argv[3]).resolve()
assert folder.is_relative_to(root/'logs') and json.loads((folder/'exit.json').read_text())['exit_code']==0
entries={}
for p in folder.rglob('*'):
 if p.is_file() and p.suffix in ('.py','.mm','.json','.log','.txt'):entries['evidence/'+str(p.relative_to(root))]=p
binary=folder/'jobs/dirty_prefix_reset_gpu';meta=json.loads((folder/'jobs/metadata.json').read_text())
assert hashlib.sha256(binary.read_bytes()).hexdigest()==meta['binary_sha256']
for name,sha in meta['source_sha256'].items():assert hashlib.sha256((folder/'jobs/source-snapshot'/name).read_bytes()).hexdigest()==sha
proof=dict(job=str(folder.relative_to(root)),binary=dict(bytes=binary.stat().st_size,sha256=meta['binary_sha256']),selected_libraries={})
for build in ('build-resident-kv-inplace','build-attention-prefix','build-global-half-alpha'):
 p=root/build/'libgarnet.dylib';proof['selected_libraries'][build]=hashlib.sha256(p.read_bytes()).hexdigest()
proof['files']={}
with tarfile.open(target,'w:gz') as archive:
 for name,p in sorted(entries.items()):
  raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
 raw=(json.dumps(proof,indent=2)+'\n').encode();info=tarfile.TarInfo('evidence/manifest.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
print(json.dumps(dict(files=len(entries),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest())))
