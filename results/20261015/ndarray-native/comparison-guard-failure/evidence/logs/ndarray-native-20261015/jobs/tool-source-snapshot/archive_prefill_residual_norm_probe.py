"""Archive terminal three-output prefill fusion probes and trained norm assets."""
import hashlib,io,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();destination=Path(sys.argv[2]);files={}
for folder in sorted((root/'logs').glob('prefill-residual-norm-*-20261014')):
 assert (folder/'exit.json').is_file(),'Job not terminal'
 for p in folder.rglob('*'):
  if p.is_file() and p.suffix in ('.py','.mm','.metal','.json','.log','.txt','.bin'):
   files['evidence/logs/'+str(p.relative_to(root/'logs'))]=p
assert files
for name in ('residual_norm_gpu.mm','run_residual_norm_probe.py','make_prefill_residual_norm_probe.py','run_prefill_residual_norm_probe.py','launch_prefill_residual_norm_probe.py','run_extra_prefill_residual_norm_probe.py','launch_extra_prefill_residual_norm_probe.py','launch_quiet_prefill_residual_norm_probe.py','run_grouped_prefill_residual_norm_probe.py','launch_grouped_prefill_residual_norm_probe.py','archive_prefill_residual_norm_probe.py'):
 files['evidence/source/'+name]=root/'source/test2026/metal'/name
manifest={}
with tarfile.open(destination,'w:gz') as tar:
 for name,p in sorted(files.items()):
  raw=p.read_bytes();manifest[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
 raw=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('evidence/files.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
print(json.dumps(dict(files=len(manifest),archive_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())))
