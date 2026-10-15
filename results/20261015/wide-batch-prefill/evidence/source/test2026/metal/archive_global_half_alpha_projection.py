"""Archive terminal trained-alpha projections and canonical omitted matrices."""
import hashlib,io,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();destination=Path(sys.argv[2]);files={};proof={};jobs={}
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for folder in sorted((root/'logs').glob('global-half-alpha-projections*-20261014')):
 status=json.loads((folder/'exit.json').read_text());jobs[folder.name]=status
 for p in folder.rglob('*'):
  if p.is_file() and p.suffix in ('.py','.mm','.h','.metal','.json','.log','.txt'):files['evidence/logs/'+str(p.relative_to(root/'logs'))]=p
 if (folder/'jobs/metadata.json').exists():
  meta=json.loads((folder/'jobs/metadata.json').read_text())
  for model,data in meta.get('assets',{}).items():
   assert digest(root/'models'/model/'model.safetensors')==data['checkpoint_sha256']
   for c in data['cases']:
    p=Path(c['path']);assert p.resolve().is_relative_to(root) and digest(p)==c['sha256'] and p.stat().st_size==c['columns']*c['inner']*2
    proof[model+':'+c['tensor']]=dict(bytes=p.stat().st_size,sha256=c['sha256'],checkpoint_sha256=data['checkpoint_sha256'])
assert jobs
for n in ('global_half_alpha_gpu.mm','run_global_half_alpha_projection.py','launch_global_half_alpha_projection.py','archive_global_half_alpha_projection.py'):
 files['evidence/source/'+n]=root/'source/test2026/metal'/n
manifest={}
with tarfile.open(destination,'w:gz') as tar:
 for name,p in files.items():
  raw=p.read_bytes();manifest[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
 for name,value in [('evidence/files.json',manifest),('evidence/omitted-matrices.json',proof),('evidence/jobs.json',jobs)]:
  raw=(json.dumps(value,indent=2)+'\n').encode();info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
print(json.dumps(dict(files=len(manifest),canonical_matrices=len(proof),archive_sha256=digest(destination))))
