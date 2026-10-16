"""Hash-checked isolated source install, preserving the prior source bytes."""
import hashlib, json, sys, tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve(); source=root/'source'
original=json.loads((root/'original.json').read_text())
candidate=json.loads((root/'manifest.json').read_text())
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert all(digest(source/n)==v for n,v in original.items()), 'Prior source identity mismatch'
backup=root/'logs/compact-joined-original-source-20261013'
backup.mkdir(exist_ok=False)
pending=[]
with tarfile.open(root/'source.tar.gz') as archive:
    members=archive.getmembers()
    assert len(members)==len(candidate) and {m.name for m in members}==set(candidate)
    assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
    for m in members:
        raw=archive.extractfile(m).read()
        assert hashlib.sha256(raw).hexdigest()==candidate[m.name]
        pending.append((m.name,raw))
for name in original:
    p=backup/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((source/name).read_bytes())
(backup/'manifest.json').write_text(json.dumps(original,indent=2))
for name,raw in pending:(source/name).write_bytes(raw)
assert all(digest(source/n)==v for n,v in candidate.items())
for old,new in [('source.tar.gz','compact-joined-source.tar.gz'),('manifest.json','compact-joined-source-manifest.json'),('original.json','compact-joined-original-manifest.json')]:
    assert not (root/new).exists();(root/old).rename(root/new)
print('Installed',len(pending),'verified compact joined source files')
