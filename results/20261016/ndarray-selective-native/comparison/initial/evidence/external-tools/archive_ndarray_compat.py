"""Preserve terminal NDArray API outcomes, including unsupported/compile failures."""
from pathlib import Path
import hashlib,io,json,sys,tarfile
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();assert not target.exists()
trial='ndarray-compat-20261015';folder=root/'logs'/trial;status=json.loads((folder/'exit.json').read_text())
assert isinstance(status['exit_code'],int);files={};omitted={}
def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
for p in folder.rglob('*'):
    if not p.is_file():continue
    name=p.relative_to(root).as_posix()
    if p.suffix in ('.json','.py','.mm','.h','.log','.txt'):files['evidence/'+name]=p
    else:omitted[name]=dict(bytes=p.stat().st_size,sha256=digest(p))
meta=folder/'jobs/metadata.json'
if meta.exists():
    for name,sha in json.loads(meta.read_text())['source_sha256'].items():assert digest(root/'source'/name)==sha,name
proof=dict(files={},omitted=omitted,jobs={trial:status},scope='Synthetic API compatibility only; no trained model or performance claim; failed modes remain failed.',full_goal_met=False)
with tarfile.open(target,'w:gz') as tar:
    for name,p in sorted(files.items()):
        raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
    raw=(json.dumps(proof,indent=2)+'\n').encode();item=tarfile.TarInfo('evidence/manifest.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
print(json.dumps(dict(files=len(files),sha256=digest(target),status=status)))
