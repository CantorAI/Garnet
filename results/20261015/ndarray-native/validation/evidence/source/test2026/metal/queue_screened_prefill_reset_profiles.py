"""Wait for exact 26-stage validation; never overlap another inference runtime."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();dependency=root/'logs/screened-prefill-reset-checks-20261015';pid=29706
assert len(sys.argv)==3 or (len(sys.argv)==4 and sys.argv[3]=='--worker')
assert json.loads((dependency/'pid.json').read_text())['pid']==pid
if not (dependency/'exit.json').exists():
    os.kill(pid,0);args=subprocess.check_output(['ps','-p',str(pid),'-o','args='],text=True)
    assert 'launch_screened_prefill_reset_checks.py' in args and str(dependency) in args and '--worker' in args
if len(sys.argv)==3:
    out.mkdir(parents=True,exist_ok=False)
    with (out/'output.log').open('w') as stream:p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(out),'--worker'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    (out/'pid.json').write_text(json.dumps(dict(pid=p.pid)));print(json.dumps(dict(pid=p.pid,output=str(out))))
else:
    start=time.monotonic();code=1
    try:
        while not (dependency/'exit.json').exists():
            os.kill(pid,0)
            if time.monotonic()-start>7200:raise TimeoutError('Exact validation dependency did not finish')
            time.sleep(5)
        status=json.loads((dependency/'exit.json').read_text());assert status['exit_code']==0
        stages=json.loads((dependency/'jobs/statuses.json').read_text());assert len(stages)==26 and all(v['exit_code']==0 for v in stages)
        manifest=json.loads((root/'screened-head-staging/manifest.json').read_text())
        for name,sha in manifest.items():assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha
        (out/'dependency.json').write_bytes((json.dumps(dict(pid=pid,status=status,completed_stages=len(stages),source_manifest=manifest),indent=2)+'\n').encode())
        snapshots=out/'helper-snapshot';snapshots.mkdir();identities={}
        for name in ('queue_screened_prefill_reset_profiles.py','run_screened_prefill_reset_profiles.py','compare_screened_prefill_reset.py','run_guarded_screened_prefill_reset_checks.py','launch_screened_prefill_reset_checks.py'):
            raw=(root/'source/test2026/metal'/name).read_bytes();identities[name]=hashlib.sha256(raw).hexdigest();(snapshots/name).write_bytes(raw)
        (out/'helper-identities.json').write_bytes((json.dumps(identities,indent=2)+'\n').encode())
        command=[sys.executable,str(root/'source/test2026/metal/run_screened_prefill_reset_profiles.py'),str(root),str(out/'native')]
        code=subprocess.call(command)
    finally:(out/'exit.json').write_bytes((json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-start))+'\n').encode())
    sys.exit(code)
