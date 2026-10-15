"""Archive both the interrupted compatibility attempt and corrected terminal run."""
import hashlib,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();destination=Path(sys.argv[2]).resolve();entries=[];omitted=[];jobs={}
def digest(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
for name,expected in [('private-bf16-probe-20261014',1),('private-bf16-retry-20261014',0),('private-bf16-confirm-20261014',1),('private-bf16-confirm-clean-20261014',-15),('private-bf16-confirm-verified-cpu-20261014',0)]:
 folder=root/'logs'/name;status=json.loads((folder/'exit.json').read_text());assert status['exit_code']==expected
 jobs[name]=status
 for p in sorted(folder.rglob('*')):
  if not p.is_file():continue
  record=dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p))
  if p.suffix in ('.log','.json','.txt','.py','.mm','.metal'):entries.append((p,record))
  else:omitted.append(record)
proof=destination.with_suffix('.manifest.json');proof.write_text(json.dumps(dict(jobs=jobs,files=[e for _,e in entries],omitted=omitted),indent=2)+'\n')
with tarfile.open(destination,'w:gz') as archive:
 archive.add(proof,arcname='manifest.json')
 for p,e in entries:archive.add(p,arcname=e['path'])
print(json.dumps(dict(files=len(entries),bytes=destination.stat().st_size,sha256=digest(destination))))
