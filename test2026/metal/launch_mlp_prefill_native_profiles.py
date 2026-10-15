"""Launch only after terminal, archived and independently audited native checks."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time

root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve()
assert len(sys.argv)==3 or (len(sys.argv)==4 and sys.argv[3]=='--worker')
checks=root/'logs/mlp-prefill-native-checks-resume-20261015'
verification=root/'mlp-prefill-native-checks-verification.json'
archive=root/'mlp-prefill-native-checks-resume-20261015.tar.gz'
assert json.loads((checks/'exit.json').read_text())['exit_code']==0
audit=json.loads(verification.read_text())
assert audit['validation_stages']==28 and audit['ctests']==8 and audit['graph_primitive_cases']==64
assert audit['screened_recovery_cases']==audit['prefix_recovery_cases']==6
assert audit['wide_reference_profiles']==4 and audit['wide_carried_steps']==12
assert hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()==audit['archive_sha256']
manifest=json.loads((root/'mlp-prefill-native-staging/manifest.json').read_text())
assert audit['source_manifest']==manifest
assert all(hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha for name,sha in manifest.items())
assert hashlib.file_digest((root/'build-mlp-prefill-graph-guardfix/libgarnet.dylib').open('rb'),'sha256').hexdigest()==audit['candidate_library_sha256']
if len(sys.argv)==3:
    out.mkdir(parents=True,exist_ok=False)
    with (out/'output.log').open('w') as stream:
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(out),'--worker'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    (out/'pid.json').write_text(json.dumps(dict(pid=p.pid)))
    print(json.dumps(dict(pid=p.pid,output=str(out))))
else:
    start=time.monotonic();code=1
    try:
        (out/'checks-verification.json').write_bytes(verification.read_bytes())
        source=root/'source';closure={}
        paths=set()
        for folder in ('src','include','cmake','xModel'):
            paths.update(p for p in (source/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.h','.cpp','.mm','.metal','.cmake'))
        paths.update(p for p in (source/'test2026/metal').glob('*') if p.is_file() and p.suffix in ('.py','.h','.cpp','.mm','.metal'))
        paths.update([source/'CMakeLists.txt',source/'test2026/pi5b/reference_decoder.py'])
        for p in sorted(paths):
            name=p.relative_to(source).as_posix();raw=p.read_bytes();closure[name]=hashlib.sha256(raw).hexdigest()
            target=out/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        (out/'source-identities.json').write_bytes((json.dumps(closure,indent=2)+'\n').encode())
        command=[sys.executable,str(source/'test2026/metal/run_mlp_prefill_native_profiles.py'),str(root),str(out/'native')]
        code=subprocess.call(command)
        assert all(hashlib.sha256((source/name).read_bytes()).hexdigest()==sha for name,sha in closure.items()),'Source changed during comparison'
    except Exception:
        code=1
        raise
    finally:
        (out/'exit.json').write_bytes((json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-start))+'\n').encode())
    sys.exit(code)
