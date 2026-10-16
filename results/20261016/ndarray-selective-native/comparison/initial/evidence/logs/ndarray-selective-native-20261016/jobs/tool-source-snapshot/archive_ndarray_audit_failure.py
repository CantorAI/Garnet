"""Preserve terminal controller audit failure and faulty helper before a narrow log repair."""
from pathlib import Path
import hashlib,io,json,sys,tarfile
root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();parent=Path(json.loads((root/'preparation.json').read_text())['parent'])
folder=root/'logs/ndarray-native-finalizer-20261015';status=json.loads((folder/'exit.json').read_text());assert status['exit_code']==1 and not target.exists()
commands=json.loads((folder/'commands.json').read_text());assert [(v['label'],v['exit_code']) for v in commands]==[('archive',0),('extract',0),('audit',1)]
assert not (root/'logs/ndarray-native-20261015').exists()
validation=root/'ndarray-native-checks-repaired-20261015.tar.gz'
def digest(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
assert digest(validation)=='f60abf2cfd3a3b1a335e096a291b5343118772a50c9ee8d5f68c37db7fd0e388'
files={'evidence/'+p.relative_to(root).as_posix():p for p in folder.rglob('*') if p.is_file()}
for name in ('finalize_ndarray_native_checks.py','finalize_shape_selected_two_checks.py','verify_ndarray_native_checks.py','archive_ndarray_native_repaired_checks.py','archive_ndarray_audit_failure.py'):
    files['evidence/consumed-tools/'+name]=parent/'source/test2026/metal'/name
proof=dict(files={},jobs={'ndarray-native-finalizer-20261015':status},validation_archive_sha256=digest(validation),
    scope='Terminal audit setup failure only. Runtime validation completed; nonempty log assumption rejected a silent compiler. No native timing launched.',full_goal_met=False)
with tarfile.open(target,'w:gz') as archive:
    for name,p in sorted(files.items()):
        raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
    raw=(json.dumps(proof,indent=2)+'\n').encode();info=tarfile.TarInfo('evidence/manifest.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
print(json.dumps(dict(files=len(files),sha256=digest(target),faulty_auditor_sha256=digest(parent/'source/test2026/metal/verify_ndarray_native_checks.py'))))
