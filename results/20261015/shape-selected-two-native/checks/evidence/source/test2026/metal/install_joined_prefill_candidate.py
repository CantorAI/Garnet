"""Install verified joined-prefill sources only in the isolated investigation."""
import hashlib,json,sys,tarfile
from pathlib import Path
r=Path(sys.argv[1]).resolve();source=r/'source'
original=json.loads((r/'joined-prefill-original-manifest.json').read_text());candidate=json.loads((r/'joined-prefill-source-manifest.json').read_text())
def digest(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
assert all(digest(source/n)==v for n,v in original.items()),'Original source identity changed'
allowed=set(original)|{'test2026/metal/joined_prefill_test.cpp','test2026/metal/run_joined_prefill_checks.py','test2026/metal/compare_joined_prefill.py','test2026/metal/run_joined_prefill_profiles.py'}
assert set(candidate)==allowed
pending=[]
with tarfile.open(r/'joined-prefill-source.tar.gz') as archive:
    members=archive.getmembers();assert len(members)==len(allowed) and {m.name for m in members}==allowed and all(m.isfile() for m in members)
    for m in members:
        raw=archive.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==candidate[m.name];pending.append((m.name,raw))
backup=r/'logs/joined-prefill-original-source-20261013';backup.mkdir(exist_ok=False)
for name in original:
    target=backup/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((source/name).read_bytes())
(backup/'manifest.json').write_text(json.dumps(original,indent=2))
for name,raw in pending:(source/name).write_bytes(raw)
assert all(digest(source/n)==v for n,v in candidate.items())
print('Preserved original files; installed',len(pending),'verified candidate sources')
