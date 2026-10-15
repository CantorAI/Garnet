"""Archive terminal native vision validation/performance, including numerical rejection."""
import hashlib,io,json,sys,tarfile
from pathlib import Path
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();files={};omitted={};jobs={}
def add(p):
    if not p.is_file():return
    key=str(p.relative_to(root))
    if p.suffix in ('.json','.py','.mm','.cpp','.h','.metal','.cmake','.log','.txt','.executed'):files['evidence/'+key]=p
    else:
        with p.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        omitted[key]=dict(bytes=p.stat().st_size,sha256=sha)
for prefix in ('parallel-vision-checks-','parallel-vision-native-'):
    for folder in sorted((root/'logs').glob(prefix+'*20261015')):
        jobs[folder.name]=json.loads((folder/'exit.json').read_text())
        for p in folder.rglob('*'):add(p)
stage=root/'parallel-vision-staging';manifest=json.loads((stage/'manifest.json').read_text());original=json.loads((stage/'original.json').read_text())
for p in stage.iterdir():add(p)
for p in (root/'logs/parallel-vision-original-source-20261015').rglob('*'):add(p)
names=set(manifest)
for job in jobs:
    meta=root/'logs'/job/'jobs/metadata.json'
    if meta.exists():names.update(json.loads(meta.read_text()).get('source_sha256',{}))
for name in names:files['evidence/source/'+name]=root/'source'/name
library=root/'build-parallel-vision-context/libgarnet.dylib';control=root/'build-global-half-alpha/libgarnet.dylib'
assert jobs
proof=dict(files={},omitted=omitted,jobs=jobs,installed=manifest,original=original,candidate_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),control_sha256=hashlib.sha256(control.read_bytes()).hexdigest())
assert proof['control_sha256']=='887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f'
with tarfile.open(target,'w:gz') as archive:
    for name,p in sorted(files.items()):
        raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());entry=tarfile.TarInfo(name);entry.size=len(raw);archive.addfile(entry,io.BytesIO(raw))
    raw=(json.dumps(proof,indent=2)+'\n').encode();entry=tarfile.TarInfo('evidence/manifest.json');entry.size=len(raw);archive.addfile(entry,io.BytesIO(raw))
print(json.dumps(dict(files=len(files),jobs=jobs,sha256=hashlib.sha256(target.read_bytes()).hexdigest())))
