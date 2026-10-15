"""Preserve original source after terminal component trials, then install exact bytes."""
import hashlib,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();stage=root/'serial-vision-staging';original=json.loads((stage/'original.json').read_text());manifest=json.loads((stage/'manifest.json').read_text())
for name in ('serial-vision-context-20261015','serial-vision-context-confirm-20261015'):assert json.loads((root/'logs'/name/'exit.json').read_text())['exit_code']==0
for name in set(manifest)-set(original):
    p=root/'source'/name
    if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest[name],'Different pre-existing investigation file: '+name
backup=root/'logs/serial-vision-original-source-20261015';backup.mkdir(exist_ok=False)
for name,sha in original.items():
    p=root/'source'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha;target=backup/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
with tarfile.open(stage/'source.tar.gz') as archive:
    assert {m.name for m in archive.getmembers()}==set(manifest)
    for item in archive.getmembers():
        p=(root/'source'/item.name).resolve();assert p.is_relative_to(root/'source') and item.isfile();raw=archive.extractfile(item).read();assert hashlib.sha256(raw).hexdigest()==manifest[item.name]
    archive.extractall(root/'source',filter='data')
assert all(hashlib.sha256((root/'source'/n).read_bytes()).hexdigest()==sha for n,sha in manifest.items())
print('Installed',len(manifest),'pinned serial vision files; originals preserved')
