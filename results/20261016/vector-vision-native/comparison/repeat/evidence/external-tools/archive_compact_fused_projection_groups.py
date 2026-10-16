"""Freeze one terminal compact-aware projection trial, preserving raw omissions."""
from pathlib import Path
import hashlib,io,json,sys,tarfile
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();trial=sys.argv[3]
assert trial in ('compact-fused-projection-groups-20261016','compact-fused-projection-groups-confirm-20261016') and not target.exists()
folder=root/'logs'/trial;status=json.loads((folder/'exit.json').read_text());assert isinstance(status['exit_code'],int)
def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
entries={};omitted={}
for p in folder.rglob('*'):
    if not p.is_file():continue
    name=p.relative_to(root).as_posix()
    if p.suffix in ('.json','.log','.txt','.py','.mm','.metal','.h'):entries['evidence/'+name]=p
    else:omitted[name]=dict(bytes=p.stat().st_size,sha256=digest(p))
proof=dict(jobs={trial:status},omitted_binaries=omitted,files={},full_goal_met=False,
 scope='One terminal component cohort; raw outputs/source retained, large binary weights hashed. No full-model gain or independent inference replication.')
with tarfile.open(target,'w:gz') as archive:
    for name,p in sorted(entries.items()):
        raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
    raw=(json.dumps(proof,indent=2)+'\n').encode();info=tarfile.TarInfo('evidence/manifest.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
print(json.dumps(dict(files=len(entries),bytes=target.stat().st_size,sha256=digest(target),status=status)))
