"""Archive terminal native-context projection trials, source and binary hashes."""
import hashlib,io,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();files={};omitted={};jobs={}
for name in (sys.argv[3:] or ['padded-mps-body-20261015']):
    folder=root/'logs'/name;status=json.loads((folder/'exit.json').read_text());assert status['exit_code']==0;jobs[name]=status
    for p in folder.rglob('*'):
        if not p.is_file():continue
        rel=str(p.relative_to(root));raw=p.read_bytes()
        if p.suffix in ('.json','.py','.mm','.h','.cpp','.metal','.log','.txt'):files['evidence/'+rel]=p
        else:omitted[rel]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
proof=dict(files={},omitted=omitted,jobs=jobs)
with tarfile.open(target,'w:gz') as archive:
    for name,p in sorted(files.items()):
        raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());item=tarfile.TarInfo(name);item.size=len(raw);archive.addfile(item,io.BytesIO(raw))
    raw=(json.dumps(proof,indent=2)+'\n').encode();item=tarfile.TarInfo('evidence/manifest.json');item.size=len(raw);archive.addfile(item,io.BytesIO(raw))
print(json.dumps(dict(files=len(files),sha256=hashlib.sha256(target.read_bytes()).hexdigest())))
