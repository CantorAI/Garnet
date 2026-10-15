"""Freeze executed global-alpha toy compatibility gate and CPU feasibility audit."""
import hashlib,io,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();destination=Path(sys.argv[2]);files={};folder=root/'logs/mps-global-alpha-toy-20261014'
assert json.loads((folder/'exit.json').read_text())['exit_code']==0
for p in folder.rglob('*'):
 if p.is_file() and p.suffix in ('.py','.mm','.metal','.json','.log','.txt'):files['evidence/logs/'+str(p.relative_to(root/'logs'))]=p
for name in ('audit_global_half_scale.py','make_mps_global_alpha_probe.py','run_mps_global_alpha_probe.py','launch_mps_global_alpha_probe.py','archive_mps_global_alpha_toy.py'):
 files['evidence/source/'+name]=root/'source/test2026/metal'/name
files['evidence/feasibility.json']=root/'logs/global-half-scale-feasibility-20261014.json';manifest={}
with tarfile.open(destination,'w:gz') as tar:
 for name,p in files.items():
  raw=p.read_bytes();manifest[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
 raw=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('evidence/files.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
print(json.dumps(dict(files=len(manifest),archive_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())))
