"""Verify and preserve every original before installing into the owned root."""
import hashlib,json,sys,tarfile
from pathlib import Path
r=Path(sys.argv[1]).resolve();assert r==Path('/Users/zsdx/garnet-metal-investigation')
s=(r/'source').resolve();assert s.is_relative_to(r)
manifest=json.loads((r/'work/direct-state-zero-manifest.json').read_text());pending=[]
assert len(manifest)==10
with tarfile.open(r/'work/direct-state-zero-source.tar.gz') as archive:
    members=archive.getmembers();assert len(members)==len(manifest) and {m.name for m in members}==set(manifest)
    for member in members:
        assert member.isfile();path=(s/member.name).resolve();assert path.is_relative_to(s)
        raw=archive.extractfile(member).read();item=manifest[member.name]
        assert hashlib.sha256(raw).hexdigest()==item['sha256']
        if item['before_sha256'] is None:assert not path.exists(),member.name
        else:assert hashlib.sha256(path.read_bytes()).hexdigest()==item['before_sha256'],member.name
        pending.append((member.name,path,raw))
backup=r/'logs/direct-state-zero-original-source-20261013';backup.mkdir(exist_ok=False)
for name,path,raw in pending:
    if path.exists():
        target=backup/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
(backup/'manifest.json').write_bytes((r/'work/direct-state-zero-manifest.json').read_bytes())
for name,path,raw in pending:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
print('Installed',len(pending),'verified files inside isolated investigation')
