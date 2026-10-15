"""Wait on exact validation, archive/audit it, then dispatch guarded native ABBA."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tarfile,time

root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve()
assert len(sys.argv)==3 or (len(sys.argv)==4 and sys.argv[3]=='--worker')
dependency=root/'logs/mlp-prefill-native-checks-resume-20261015';pid=73896
assert json.loads((dependency/'pid.json').read_text())['pid']==pid
if len(sys.argv)==3:
    assert not (root/'logs/mlp-prefill-native-20261015').exists()
    out.mkdir(parents=True,exist_ok=False)
    with (out/'output.log').open('w') as stream:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(out),'--worker'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    (out/'pid.json').write_text(json.dumps(dict(pid=process.pid)))
    print(json.dumps(dict(pid=process.pid,output=str(out))))
else:
    started=time.monotonic();code=1;state='waiting_for_exact_validation';steps=[]
    def run(name,command):
        print('Starting',name,flush=True)
        with (out/(name+'.log')).open('w') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        steps.append(dict(name=name,command=command,exit_code=result.returncode))
        (out/'steps.json').write_text(json.dumps(steps,indent=2)+'\n')
        assert result.returncode==0,name
        print('Completed',name,flush=True)
    try:
        manifest=json.loads((root/'mlp-prefill-native-staging/manifest.json').read_text())
        assert len(manifest)==18
        while not (dependency/'exit.json').exists():
            os.kill(pid,0)
            args=subprocess.check_output(['/bin/ps','-p',str(pid),'-o','args='],text=True)
            assert 'launch_resume_mlp_prefill_native_checks.py' in args and str(dependency) in args and '--worker' in args
            if time.monotonic()-started>7200:raise TimeoutError('Exact validation did not finish; no restart')
            time.sleep(5)
        assert json.loads((dependency/'exit.json').read_text())['exit_code']==0,'Validation failed; keep evidence and do not benchmark'
        assert all(hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha for name,sha in manifest.items())
        state='archiving_validation';archive=root/'mlp-prefill-native-checks-resume-20261015.tar.gz'
        assert not archive.exists()
        scripts=root/'source/test2026/metal'
        run('archive-validation',[sys.executable,str(scripts/'archive_mlp_prefill_native_checks.py'),str(root),str(archive)])
        state='auditing_validation';audit=root/'validation-audits/mlp-prefill-native-checks-resume-20261015'
        audit.mkdir(parents=True,exist_ok=False)
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                assert member.isfile();p=(audit/member.name).resolve();assert p.is_relative_to(audit)
                p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tar.extractfile(member).read())
        verification=root/'mlp-prefill-native-checks-verification.json'
        assert not verification.exists()
        run('audit-validation',[sys.executable,str(scripts/'verify_mlp_prefill_native_checks.py'),str(audit),str(verification),str(archive)])
        checked=json.loads(verification.read_text())
        assert checked['validation_stages']==28 and checked['ctests']==8 and checked['graph_primitive_cases']==64
        state='dispatching_native_benchmark';native=root/'logs/mlp-prefill-native-20261015'
        run('launch-native-profiles',[sys.executable,str(scripts/'launch_mlp_prefill_native_profiles.py'),str(root),str(native)])
        dispatched=json.loads((native/'pid.json').read_text());os.kill(dispatched['pid'],0)
        (out/'benchmark-dispatch.json').write_text(json.dumps(dict(**dispatched,output=str(native),validation_archive_sha256=checked['archive_sha256']),indent=2)+'\n')
        state='native_benchmark_dispatched';code=0
    finally:
        (out/'exit.json').write_text(json.dumps(dict(exit_code=code,state=state,wall_seconds=time.monotonic()-started,scope='Validation finalization/native benchmark dispatch only; no performance result or goal-completion claim'))+'\n')
    sys.exit(code)
