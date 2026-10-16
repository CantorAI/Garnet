"""Wait for the exact owned full-validation handle, then recover and benchmark."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();dependency=root/'logs/fused-head-argmax-checks-20261015'
assert len(sys.argv)==3 or (len(sys.argv)==4 and sys.argv[3]=='--worker')
pid=68935;assert json.loads((dependency/'pid.json').read_text())['pid']==pid
if not (dependency/'exit.json').exists():
    os.kill(pid,0);args=subprocess.check_output(['ps','-p',str(pid),'-o','args='],text=True).strip()
    assert 'launch_fused_head_argmax_checks.py' in args and str(dependency) in args and '--worker' in args
if len(sys.argv)==3:
    out.mkdir(parents=True,exist_ok=False)
    with (out/'output.log').open('w') as stream:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(out),'--worker'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    (out/'pid.json').write_text(json.dumps(dict(pid=process.pid)));print(json.dumps(dict(pid=process.pid,output=str(out))))
else:
    begin=time.monotonic();code=1
    try:
        while not (dependency/'exit.json').exists():
            os.kill(pid,0)
            if time.monotonic()-begin>7200:raise TimeoutError('Owned validation dependency wait expired')
            time.sleep(5)
        status=json.loads((dependency/'exit.json').read_text());assert status['exit_code']==0
        stages=json.loads((dependency/'jobs/statuses.json').read_text());assert len(stages)==19 and all(v['exit_code']==0 for v in stages)
        pinned=json.loads((root/'fused-head-argmax-staging/manifest.json').read_text())
        for name,sha in pinned.items():assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha
        (out/'dependency.json').write_bytes((json.dumps(dict(pid=pid,status=status,completed_original_stages=len(stages),source_manifest=pinned),indent=2)+'\n').encode())
        helpers=('queue_fused_head_argmax_profiles.py','run_fused_head_session_checks.py','test_fused_head_session_reference.py','run_fused_head_session_reference.py','run_fused_head_argmax_profiles.py','compare_fused_head_argmax.py')
        snapshots=out/'helper-snapshot';snapshots.mkdir();identities={}
        for name in helpers:
            data=(root/'source/test2026/metal'/name).read_bytes();identities[name]=hashlib.sha256(data).hexdigest();(snapshots/name).write_bytes(data)
        (out/'helper-identities.json').write_bytes((json.dumps(identities,indent=2)+'\n').encode())
        for phase,helper in [('recovery','run_fused_head_session_checks.py'),('native','run_fused_head_argmax_profiles.py')]:
            start=time.monotonic();command=[sys.executable,str(root/'source/test2026/metal'/helper),str(root),str(out/phase)]
            code=subprocess.call(command);(out/(phase+'-exit.json')).write_bytes((json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-start,command=command),indent=2)+'\n').encode())
            if code:break
    finally:(out/'exit.json').write_bytes((json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-begin))+'\n').encode())
    sys.exit(code)
