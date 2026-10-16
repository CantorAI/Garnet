"""Freeze the missing-helper failure; runtime archives remain separate unchanged."""
from pathlib import Path
import hashlib,io,json,sys,tarfile
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();assert not target.exists()
folders=[root/'shape-selected-two-investigation-20261015/logs/shape-native-bf16-finalize-20261015',
         root/'logs/queue-bf16-feed-mlp-component-20261015']
files={};jobs={}
for folder in folders:
    status=json.loads((folder/'exit.json').read_text());assert status['exit_code']==1
    jobs[folder.relative_to(root).as_posix()]=status
    for p in folder.rglob('*'):
        if p.is_file():files['evidence/'+p.relative_to(root).as_posix()]=p
failure=(folders[0]/'bf16-audit.log').read_text()
assert 'FileNotFoundError' in failure and 'verify_mlp_command_buffer_primitive.py' in failure
proof=dict(files={},jobs=jobs,
    native_runtime_archive_sha256='f18be8192eca6f2f7cd51590db9ceae68b2e70e7d3ce602e75216a02edb501eb',
    primitive_runtime_archive_sha256='914fce255ea3d8b17e2ed4e7034ced808b57438df03a5904b33c1cf556573f1b',
    scope='Controller/auditor setup failure only. Native and primitive runtime executions succeeded and their separate archives remain unchanged. Missing auditor dependency was copied only after terminal archival.',full_goal_met=False)
with tarfile.open(target,'w:gz') as tar:
    for name,p in sorted(files.items()):
        raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
    raw=(json.dumps(proof,indent=2)+'\n').encode();item=tarfile.TarInfo('evidence/manifest.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
with target.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
print(json.dumps(dict(files=len(files),sha256=sha)))
