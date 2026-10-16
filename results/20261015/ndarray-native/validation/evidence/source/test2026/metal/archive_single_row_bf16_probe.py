"""Archive terminal single-row probes and canonical trained-matrix proofs."""
import hashlib,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();destination=Path(sys.argv[2]).resolve();entries=[];omitted=[];jobs={};matrices={}
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for folder in sorted((root/'logs').glob('single-row-bf16-*-20261014')):
 status=json.loads((folder/'exit.json').read_text());jobs[folder.name]=status
 for p in sorted(folder.rglob('*')):
  if not p.is_file():continue
  record=dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p))
  if p.suffix in ('.log','.json','.txt','.py','.mm','.metal'):entries.append((p,record))
  else:omitted.append(record)
 if (folder/'jobs/metadata.json').is_file():
  meta=json.loads((folder/'jobs/metadata.json').read_text())
  for model,assets in meta.get('assets',{}).items():
   assert digest(root/'models'/model/'model.safetensors')==assets['checkpoint_sha256']
   for item in assets['cases']:
    p=root/'logs/mps-gemv-probe-20261013/jobs'/(model+'-assets')/(item['name']+'.bin')
    assert digest(p)==item['sha256'] and p.stat().st_size==item['columns']*item['inner']*2
    matrices[model+':'+item['name']]=dict(bytes=p.stat().st_size,sha256=item['sha256'],checkpoint_sha256=assets['checkpoint_sha256'])
assert jobs
proof=destination.with_suffix('.manifest.json');proof.write_text(json.dumps(dict(jobs=jobs,files=[e for _,e in entries],omitted=omitted,omitted_matrices=matrices),indent=2)+'\n')
with tarfile.open(destination,'w:gz') as tar:
 tar.add(proof,arcname='manifest.json')
 for p,e in entries:tar.add(p,arcname=e['path'])
print(json.dumps(dict(files=len(entries),canonical_matrix_proofs=len(matrices),archive_sha256=digest(destination))))
