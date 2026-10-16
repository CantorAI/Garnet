"""Freeze completed paired-product trials and canonical trained-member proofs."""
import hashlib,io,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();destination=Path(sys.argv[2]);files={};proof={};jobs={}
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for folder in sorted((root/'logs').glob('joined-global-alpha-probe*-20261014')):
 status=json.loads((folder/'exit.json').read_text());assert status['exit_code']==0;jobs[folder.name]=status
 for p in folder.rglob('*'):
  if p.is_file() and p.suffix in ('.py','.mm','.h','.json','.log','.txt'):files['evidence/logs/'+str(p.relative_to(root/'logs'))]=p
 meta=json.loads((folder/'jobs/metadata.json').read_text())
 for model,data in meta['assets'].items():
  assert digest(root/'models'/model/'model.safetensors')==data['checkpoint_sha256']
  joined=data['joined'];p=Path(joined['path']);assert p.resolve().is_relative_to(root) and digest(p)==joined['sha256'] and p.stat().st_size==joined['columns']*joined['inner']*2
  h=hashlib.sha256()
  for member in data['members']:
   original=root/'logs/mps-gemv-probe-20261013/jobs'/(model+'-assets')/(member['name']+'.bin')
   assert digest(original)==member['sha256'];h.update(original.read_bytes())
  assert h.hexdigest()==joined['sha256']
  proof[model]=dict(checkpoint_sha256=data['checkpoint_sha256'],joined_sha256=joined['sha256'],bytes=p.stat().st_size,members=data['members'])
assert jobs
for name in ('joined_global_alpha_gpu.mm','run_joined_global_alpha_probe.py','launch_joined_global_alpha_probe.py','archive_joined_global_alpha_probe.py'):
 files['evidence/source/'+name]=root/'source/test2026/metal'/name
manifest={}
with tarfile.open(destination,'w:gz') as tar:
 for name,p in files.items():
  raw=p.read_bytes();manifest[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
 for name,value in [('evidence/files.json',manifest),('evidence/omitted-matrices.json',proof),('evidence/jobs.json',jobs)]:
  raw=(json.dumps(value,indent=2)+'\n').encode();info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
print(json.dumps(dict(files=len(manifest),paired_matrix_proofs=len(proof),sha256=digest(destination))))
