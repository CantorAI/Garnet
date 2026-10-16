"""Freeze both terminal CPU union audits; model payloads omitted by full identity."""
from pathlib import Path
import hashlib,io,json,sys,tarfile
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();assert not target.exists();entries={};omitted={};jobs={}
def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
for trial in ('joined-union-half-admission-20261016','joined-union-half-admission-cpp-20261016'):
    folder=root/'logs'/trial;jobs[trial]=json.loads((folder/'exit.json').read_text());assert jobs[trial]['exit_code']==0
    for p in folder.rglob('*'):
        if not p.is_file():continue
        name=p.relative_to(root).as_posix()
        if p.suffix in ('.json','.log','.py','.mm','.h','.txt'):entries['evidence/'+name]=p
        else:omitted[name]=dict(bytes=p.stat().st_size,sha256=digest(p))
report=json.loads((root/'logs/joined-union-half-admission-20261016/report.json').read_text())
for model in report['models']:
    p=Path(model['checkpoint']);assert digest(p)==model['checkpoint_sha256'];omitted[p.relative_to(root).as_posix()]=dict(bytes=p.stat().st_size,sha256=model['checkpoint_sha256'])
for name in ('archive_joined_union_half_admission.py','verify_joined_union_half_admission.py'):
    p=Path(__file__).with_name(name);entries['evidence/tools/'+name]=p
proof=dict(files={},omitted=omitted,jobs=jobs,scope='CPU exact-scale full-layer eligibility and original-codec roundtrips only; no native/GPU/inference/performance claim.',full_goal_met=False)
with tarfile.open(target,'w:gz') as archive:
    for name,p in sorted(entries.items()):
        raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
    raw=(json.dumps(proof,indent=2)+'\n').encode();info=tarfile.TarInfo('evidence/manifest.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
print(json.dumps(dict(files=len(entries),sha256=digest(target),jobs=jobs)))
