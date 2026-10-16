"""Freeze terminal synthetic vector probe without unrelated model assets or library assumptions."""
from pathlib import Path
import hashlib,io,json,sys,tarfile
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();assert not target.exists()
trials=sys.argv[3:] or ['vector-vision-context-20261016'];files={};omitted={};jobs={}
def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
for trial in trials:
    assert trial in ('vector-vision-context-20261016','vector-vision-context-confirm-20261016')
    folder=root/'logs'/trial;status=json.loads((folder/'exit.json').read_text());assert isinstance(status['exit_code'],int);jobs[trial]=status
    for p in folder.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(root).as_posix()
        if p.suffix in ('.json','.py','.mm','.h','.metal','.cpp','.log','.txt'):files['evidence/'+rel]=p
        else:omitted[rel]=dict(bytes=p.stat().st_size,sha256=digest(p))
    meta=folder/'jobs/metadata.json'
    if meta.exists():
        identities=json.loads(meta.read_text())['source_sha256']
        for name,sha in identities.items():assert digest(root/'source'/name)==sha
for name in ('archive_vector_vision_component.py','launch_vector_vision_context.py','verify_vector_vision_context.py','analyze_vector_vision_context.py'):
    p=Path(__file__).with_name(name);files['evidence/tools/'+name]=p
proof=dict(files={},omitted=omitted,jobs=jobs,scope='Synthetic component only; exact scalar control, vector candidates and null. Terminal failures preserved. No model performance claim.',full_goal_met=False)
with tarfile.open(target,'w:gz') as tar:
    for name,p in sorted(files.items()):
        raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());member=tarfile.TarInfo(name);member.size=len(raw);tar.addfile(member,io.BytesIO(raw))
    raw=(json.dumps(proof,indent=2)+'\n').encode();member=tarfile.TarInfo('evidence/manifest.json');member.size=len(raw);tar.addfile(member,io.BytesIO(raw))
print(json.dumps(dict(files=len(files),sha256=digest(target),jobs=jobs)))
