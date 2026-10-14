"""Install only hash-verified owned investigation sources, retaining original bytes."""
import hashlib,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();staging=root/'constant-head-staging';source=root/'source'
original=json.loads((staging/'original.json').read_text());candidate=json.loads((staging/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert all(digest(source/n)==v for n,v in original.items()),'Prior source identity changed'
backup=root/'logs/constant-head-original-source-20261014';backup.mkdir(exist_ok=False);pending=[]
with tarfile.open(staging/'source.tar.gz') as archive:
 members=archive.getmembers();assert len(members)==len(candidate) and {m.name for m in members}==set(candidate)
 assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
 for m in members:
  data=archive.extractfile(m).read();assert hashlib.sha256(data).hexdigest()==candidate[m.name];pending.append((m.name,data))
for n in original:
 p=backup/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((source/n).read_bytes())
(backup/'manifest.json').write_text(json.dumps(original,indent=2))
for n,data in pending:(source/n).write_bytes(data)
assert all(digest(source/n)==v for n,v in candidate.items())
print('Installed',len(pending),'verified constant-head sources; preserved original bytes')
