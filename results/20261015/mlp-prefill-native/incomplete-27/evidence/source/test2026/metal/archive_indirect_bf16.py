"""Archive terminal logs, executed source identities and cached fixture proofs."""
import hashlib,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();job=root/'logs/indirect-bf16-projection-20261014';destination=Path(sys.argv[2]).resolve()
assert json.loads((job/'exit.json').read_text())['exit_code']==0
entries=[];omitted=[]
for p in sorted(job.rglob('*')):
 if not p.is_file():continue
 record=dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest())
 if p.suffix in ('.log','.json','.txt','.py','.mm','.metal'):entries.append((p,record))
 else:omitted.append(record)
metadata=json.loads((job/'jobs/metadata.json').read_text())
assert metadata['binary_sha256']==next(x['sha256'] for x in omitted if x['path'].endswith('/indirect_bf16_gpu'))
proof=destination.with_suffix('.manifest.json');proof.write_text(json.dumps(dict(files=[x for _,x in entries],omitted=omitted),indent=2)+'\n')
with tarfile.open(destination,'w:gz') as archive:
 archive.add(proof,arcname='manifest.json')
 for p,record in entries:archive.add(p,arcname=record['path'])
print(json.dumps(dict(files=len(entries),bytes=destination.stat().st_size,sha256=hashlib.file_digest(destination.open('rb'),'sha256').hexdigest())))
