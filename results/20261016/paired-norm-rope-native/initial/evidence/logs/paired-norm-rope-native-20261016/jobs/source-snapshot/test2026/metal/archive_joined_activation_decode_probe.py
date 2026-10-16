"""Preserve terminal joined-input trials including failed admission and raw source."""
import hashlib,io,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();entries={};jobs={};omitted={}
for folder in sorted((root/'logs').glob('joined-activation-decode-*20261014')):
 status=json.loads((folder/'exit.json').read_text());jobs[folder.name]=status
 for p in folder.rglob('*'):
  if p.is_file() and p.suffix in ('.json','.log','.txt','.py','.mm','.metal','.h'):entries['evidence/'+str(p.relative_to(root))]=p
 binary=folder/'jobs/joined_activation_decode_gpu'
 if binary.exists():omitted[str(binary.relative_to(root))]=dict(bytes=binary.stat().st_size,sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
assert jobs
proof=dict(jobs=jobs,omitted_binaries=omitted,files={})
with tarfile.open(target,'w:gz') as archive:
 for name,p in sorted(entries.items()):
  raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
 raw=(json.dumps(proof,indent=2)+'\n').encode();info=tarfile.TarInfo('evidence/manifest.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
print(json.dumps(dict(files=len(entries),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest())))
