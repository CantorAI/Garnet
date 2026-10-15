"""Preserve terminal screening trials, exact source and all omitted-data hashes."""
import hashlib,io,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();files={};omitted={};jobs={}
for folder in sorted((root/'logs').glob('screened-head-*20261015')):
    jobs[folder.name]=json.loads((folder/'exit.json').read_text())
    for p in folder.rglob('*'):
        if not p.is_file():continue
        name=str(p.relative_to(root));raw=p.read_bytes()
        if p.suffix in ('.json','.py','.mm','.metal','.log','.txt'):files['evidence/'+name]=p
        else:omitted[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
assert jobs;proof=dict(files={},omitted=omitted,jobs=jobs)
with tarfile.open(target,'w:gz') as archive:
    for name,p in sorted(files.items()):
        raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());entry=tarfile.TarInfo(name);entry.size=len(raw);archive.addfile(entry,io.BytesIO(raw))
    raw=(json.dumps(proof,indent=2)+'\n').encode();entry=tarfile.TarInfo('evidence/manifest.json');entry.size=len(raw);archive.addfile(entry,io.BytesIO(raw))
print(json.dumps(dict(files=len(files),omitted=len(omitted),jobs=jobs,sha256=hashlib.sha256(target.read_bytes()).hexdigest())))
