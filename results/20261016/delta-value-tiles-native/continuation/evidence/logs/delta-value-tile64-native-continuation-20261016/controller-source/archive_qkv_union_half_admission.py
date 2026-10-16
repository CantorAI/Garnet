"""Freeze terminal QKV CPU audit, launch provenance and original codec transformation."""
from pathlib import Path
import hashlib,io,json,sys,tarfile
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();folder=root/'logs/qkv-union-half-admission-20261016';assert not target.exists()
status=json.loads((folder/'exit.json').read_text());assert isinstance(status['exit_code'],int)
def digest(p):
 with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
files={};omitted={}
for p in folder.rglob('*'):
 if not p.is_file():continue
 name=p.relative_to(root).as_posix()
 if p.suffix in ('.json','.log','.py','.mm','.h','.txt'):files['evidence/'+name]=p
 else:omitted[name]=dict(bytes=p.stat().st_size,sha256=digest(p))
for suffix in ('json','log'):
 p=root/'work'/('qkv-union-half-admission-20261016.launch.'+suffix);files['evidence/work/'+p.name]=p
for name in ('archive_qkv_union_half_admission.py','verify_qkv_union_half_admission.py'):
 p=Path(__file__).with_name(name);files['evidence/tools/'+name]=p
report=json.loads((folder/'report.json').read_text())
for model in report['models']:
 p=Path(model['checkpoint']);assert digest(p)==model['checkpoint_sha256'];omitted[p.relative_to(root).as_posix()]=dict(bytes=p.stat().st_size,sha256=model['checkpoint_sha256'])
proof=dict(files={},omitted=omitted,jobs={folder.name:status},scope='CPU full learned separate QKV common-scale admission only; no GPU arithmetic, graph/liveness, speed or model claim.',full_goal_met=False)
with tarfile.open(target,'w:gz') as tar:
 for name,p in sorted(files.items()):
  raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
 raw=(json.dumps(proof,indent=2)+'\n').encode();item=tarfile.TarInfo('evidence/manifest.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
print(json.dumps(dict(files=len(files),sha256=digest(target),status=status)))
