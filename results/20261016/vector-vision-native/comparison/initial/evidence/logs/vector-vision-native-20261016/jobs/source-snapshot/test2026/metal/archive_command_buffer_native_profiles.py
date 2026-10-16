"""Freeze native outputs, both source closures and original trained gates."""
from pathlib import Path
import hashlib,io,json,subprocess,sys,tarfile

p=Path(__file__).with_name('archive_command_buffer_native_checks.py')
source=p.read_text();source=source[:source.index('proof=dict(')]
exec(compile(source,str(p),'exec'))
native_trial=sys.argv[3] if len(sys.argv)>3 else 'mlp-command-buffer-native-20261015'
assert native_trial in ('mlp-command-buffer-native-20261015','mlp-command-buffer-native-retry-20261015')
native=root/'logs'/native_trial;native_status=json.loads((native/'exit.json').read_text())
assert isinstance(native_status['exit_code'],int)
tree(native)
identities=json.loads((native/'source-identities.json').read_text())
for name,sha in identities.items():
    assert digest(root/'source'/name)==sha and digest(native/'source-snapshot'/name)==sha
    add(root/'source'/name)
external=json.loads((native/'external-tool-identities.json').read_text())
for name,sha in external['files'].items():
    path=Path(external['directory'])/name
    assert digest(path)==sha and digest(native/'tool-source-snapshot'/name)==sha
    files['evidence/external-tools/'+name]=path
audit=json.loads((native/'checks-verification.json').read_text())
assert digest(root/'command-buffer-native-checks-20261015.tar.gz')==audit['archive_sha256']
hardware={name:subprocess.check_output(['sysctl','-n',name],text=True).strip() for name in ('machdep.cpu.brand_string','hw.memsize','hw.model')}
hardware['sw_vers']=subprocess.check_output(['sw_vers'],text=True)
proof=dict(files={},omitted=omitted,hardware=hardware,jobs={trial:status,native_trial:native_status},
    library_sha256=libraries,checks_archive_sha256=audit['archive_sha256'],
    scope='Complete native output/timing cohort, command-buffer graph versus retained c186; no fresh vLLM ratio',full_goal_met=False)
assert not target.exists()
with tarfile.open(target,'w:gz') as tar:
    for name,path in sorted(files.items()):
        raw=path.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
    raw=(json.dumps(proof,indent=2)+'\n').encode()
    item=tarfile.TarInfo('evidence/manifest.json');item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
print(json.dumps(dict(files=len(files),sha256=digest(target),status=native_status)))
