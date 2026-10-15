"""Archive terminal complete-model MLP comparison and its original native gates."""
from pathlib import Path
import hashlib,io,json,subprocess,sys,tarfile

p=Path(__file__).with_name('archive_mlp_prefill_native_checks.py')
source=p.read_text();source=source[:source.index('proof=dict(')]
source=source.replace("trial=sys.argv[3] if len(sys.argv)>3 else 'mlp-prefill-native-checks-resume-20261015'", "trial='mlp-prefill-native-checks-resume-20261015'")
exec(compile(source,str(p),'exec'))
native_trial=sys.argv[3] if len(sys.argv)>3 else 'mlp-prefill-native-20261015'
native=root/'logs'/native_trial
native_status=json.loads((native/'exit.json').read_text())
assert isinstance(native_status['exit_code'],int)
for p in native.rglob('*'):
    if p.is_file():add(p)
identities=json.loads((native/'source-identities.json').read_text())
for name,sha in identities.items():
    p=root/'source'/name;assert digest(p)==sha,name
    assert digest(native/'source-snapshot'/name)==sha
    add(p)
for name in ('archive_mlp_prefill_native_profiles.py','verify_mlp_prefill_native_profiles.py'):
    p=root/'source/test2026/metal'/name
    if p.is_file():add(p)
hardware={name:subprocess.check_output(['sysctl','-n',name],text=True).strip() for name in ('machdep.cpu.brand_string','hw.memsize','hw.model')}
hardware['sw_vers']=subprocess.check_output(['sw_vers'],text=True)
audit=json.loads((native/'checks-verification.json').read_text())
assert digest(root/'mlp-prefill-native-checks-resume-20261015.tar.gz')==audit['archive_sha256']
proof=dict(files={},omitted=omitted,hardware=hardware,jobs={trial:status,native_trial:native_status},
           library_sha256={name:omitted[name+'/libgarnet.dylib']['sha256'] for name in ('build-coalesced-bf16-body','build-mlp-prefill-graph-guardfix')},
           checks_archive_sha256=audit['archive_sha256'],
           scope='Complete native control/candidate comparison; no fresh vLLM measurement',full_goal_met=False)
with tarfile.open(target,'w:gz') as archive:
    for name,p in sorted(files.items()):
        raw=p.read_bytes();proof['files'][name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
    raw=(json.dumps(proof,indent=2)+'\n').encode();info=tarfile.TarInfo('evidence/manifest.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
print(json.dumps(dict(files=len(files),sha256=digest(target),status=native_status)))
