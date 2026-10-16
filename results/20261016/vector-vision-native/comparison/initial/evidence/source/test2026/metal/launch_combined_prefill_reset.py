"""Durable sequential full validation and serving; preserve every existing build."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve()
assert len(sys.argv)==3 or (len(sys.argv)==4 and sys.argv[3]=='--worker')
assert json.loads((root/'logs/prefill-stream-confirm-native-20261015/exit.json').read_text())['exit_code']==0
assert hashlib.file_digest((root/'build-prefill-stream/libgarnet.dylib').open('rb'),'sha256').hexdigest()=='bad8c5db120df0e6ac107a0fabc6b19708a379f36785b7b5c8869a30fd14534b'
manifest=json.loads((root/'prefill-stream-staging/manifest.json').read_text())
assert all(hashlib.file_digest((root/'source'/name).open('rb'),'sha256').hexdigest()==sha for name,sha in manifest.items())
if len(sys.argv)==3:
    assert not (root/'build-combined-prefill-reset').exists()
    out.mkdir(parents=True,exist_ok=False)
    with (out/'output.log').open('w') as stream:
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(out),'--worker'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    (out/'pid.json').write_text(json.dumps(dict(pid=p.pid)))
    print(json.dumps(dict(pid=p.pid,output=str(out))))
else:
    begin=time.monotonic();code=1
    try:
        helpers=('launch_combined_prefill_reset.py','run_guarded_combined_prefill_reset_checks.py','run_combined_prefill_reset_profiles.py','compare_combined_prefill_reset.py')
        identities={}
        for name in helpers:
            raw=(root/'source/test2026/metal'/name).read_bytes()
            identities[name]=hashlib.sha256(raw).hexdigest()
            (out/(name+'.executed')).write_bytes(raw)
        (out/'source-identities.json').write_text(json.dumps(dict(manifest=manifest,helpers=identities),indent=2))
        for phase,script in [('checks','run_guarded_combined_prefill_reset_checks.py'),('native','run_combined_prefill_reset_profiles.py')]:
            start=time.monotonic()
            command=[sys.executable,str(root/'source/test2026/metal'/script),str(root),str(out/phase)]
            code=1
            code=subprocess.call(command)
            (out/(phase+'-exit.json')).write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-start,command=command)))
            if code:break
    finally:
        (out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-begin)))
    sys.exit(code)
