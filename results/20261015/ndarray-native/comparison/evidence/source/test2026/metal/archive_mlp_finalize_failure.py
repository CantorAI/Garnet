"""Freeze stopped controller/auditor versions before repairing missing coverage."""
from pathlib import Path
import hashlib,io,json,sys,tarfile

root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();files={}
for name in ('mlp-prefill-native-finalize-20261015','mlp-prefill-native-and-eight-finalize-20261015'):
    folder=root/'logs'/name;assert json.loads((folder/'exit.json').read_text())['exit_code']==1
    for p in folder.rglob('*'):
        if p.is_file():files[p.relative_to(root).as_posix()]=p
for name in ('finalize_mlp_checks_and_launch_profiles.py','finalize_mlp_profiles_and_launch_eight.py',
             'verify_mlp_prefill_native_checks.py','archive_mlp_prefill_native_checks.py',
             'launch_mlp_prefill_native_profiles.py','archive_mlp_prefill_native_profiles.py'):
    p=root/'source/test2026/metal'/name;files[p.relative_to(root).as_posix()]=p
proof=dict(files={},scope='Terminal controller audit failure: original resumed coverage27, required28; benchmark not dispatched',full_goal_met=False)
with tarfile.open(target,'w:gz') as tar:
    for name,p in sorted(files.items()):
        raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        item=tarfile.TarInfo('evidence/'+name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
    raw=(json.dumps(proof,indent=2)+'\n').encode();item=tarfile.TarInfo('evidence/manifest.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
with target.open('rb') as stream:print(json.dumps(dict(files=len(files),sha256=hashlib.file_digest(stream,'sha256').hexdigest())))
