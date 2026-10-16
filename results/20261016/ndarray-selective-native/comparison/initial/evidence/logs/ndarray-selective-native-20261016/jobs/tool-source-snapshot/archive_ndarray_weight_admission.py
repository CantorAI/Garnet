"""Freeze terminal CPU checkpoint admission and original-codec cross-check; omit raw weights by identity."""
from pathlib import Path
import hashlib,io,json,sys,tarfile
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();assert not target.exists()
trials=('ndarray-weight-admission-20261016','ndarray-weight-admission-cpp-20261016');files={};omitted={};jobs={}
def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
for trial in trials:
    folder=root/'logs'/trial;jobs[trial]=json.loads((folder/'exit.json').read_text());assert jobs[trial]['exit_code']==0
    for p in folder.rglob('*'):
        if not p.is_file():continue
        name=p.relative_to(root).as_posix()
        if p.suffix in ('.json','.py','.h','.mm','.log','.txt'):files['evidence/'+name]=p
        else:omitted[name]=dict(bytes=p.stat().st_size,sha256=digest(p))
report=json.loads((root/'logs'/trials[0]/'report.json').read_text())
for m in report['models']:
    p=root/'models'/m['model']/'model.safetensors';assert digest(p)==m['checkpoint_sha256'];omitted[p.relative_to(root).as_posix()]=dict(bytes=p.stat().st_size,sha256=m['checkpoint_sha256'])
meta=json.loads((root/'logs'/trials[0]/'metadata.json').read_text())
for name,sha in meta['source_sha256'].items():assert digest(Path(name))==sha
for name in ('archive_ndarray_weight_admission.py',):
    p=Path(__file__).with_name(name);files['evidence/tools/'+name]=p
proof=dict(files={},omitted=omitted,jobs=jobs,scope='CPU codec eligibility and bitwise weight roundtrips only; no GPU/inference/performance claim.',full_goal_met=False)
with tarfile.open(target,'w:gz') as archive:
    for name,p in sorted(files.items()):
        raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
    raw=(json.dumps(proof,indent=2)+'\n').encode();info=tarfile.TarInfo('evidence/manifest.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
print(json.dumps(dict(files=len(files),sha256=digest(target),jobs=jobs)))
