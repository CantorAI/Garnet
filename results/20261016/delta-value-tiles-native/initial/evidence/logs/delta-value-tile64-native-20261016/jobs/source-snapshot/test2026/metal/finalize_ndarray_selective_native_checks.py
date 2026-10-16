"""Wait the exact validation PID, freeze terminal evidence, and audit success without rerunning GPU work."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();expected_pid=int(sys.argv[3])
assert root.name=='ndarray-selective-native-packed-investigation-20261016'
if len(sys.argv)==4:
    out.mkdir(parents=True,exist_ok=False)
    with (out/'output.log').open('w') as stream:
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(out),str(expected_pid),'--worker'],
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    (out/'pid.json').write_text(json.dumps(dict(pid=p.pid)));print(json.dumps(dict(pid=p.pid,output=str(out))))
else:
    assert len(sys.argv)==5 and sys.argv[4]=='--worker'
    begin=time.monotonic();code=1
    def digest(p):
        with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    try:
        trial='ndarray-selective-native-packed-checks-20261016';checks=root/'logs'/trial
        assert json.loads((checks/'pid.json').read_text())['pid']==expected_pid
        parent=Path(json.loads((root/'preparation.json').read_text())['parent']);tools=Path(__file__).parent
        helpers=[Path(__file__),tools/'archive_ndarray_selective_native_checks.py',tools/'verify_ndarray_selective_native_checks.py',parent/'work/freeze_mlp_evidence.py']
        identities={str(p):digest(p) for p in helpers}
        for i,p in enumerate(helpers):
            target=out/'controller-source'/str(i)/p.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
        (out/'metadata.json').write_text(json.dumps(dict(expected_validation_pid=expected_pid,source_sha256=identities,
            command=sys.argv,scope=__doc__,full_goal_met=False),indent=2))
        while True:
            process=subprocess.run(['ps','-p',str(expected_pid),'-o','command='],capture_output=True,text=True)
            live=process.returncode==0 and bool(process.stdout.strip())
            if live:assert 'launch_ndarray_selective_native_checks.py' in process.stdout and str(root) in process.stdout
            (out/'observation.json').write_text(json.dumps(dict(validation_pid=expected_pid,live=live,
                terminal_status_present=(checks/'exit.json').exists(),elapsed_seconds=time.monotonic()-begin)))
            if not live:
                assert (checks/'exit.json').exists(),'Exact process missing without terminal status; do not restart validation'
                break
            time.sleep(15)
        status=json.loads((checks/'exit.json').read_text());assert isinstance(status['exit_code'],int)
        archive=parent/'ndarray-selective-native-packed-checks-20261016.tar.gz';assert not archive.exists()
        command=[sys.executable,str(tools/'archive_ndarray_selective_native_checks.py'),str(root),str(archive)]
        p=subprocess.run(command,capture_output=True,text=True)
        (out/'archive.log').write_text(p.stdout+p.stderr);assert p.returncode==0
        identity=json.loads(p.stdout.splitlines()[-1]);assert identity['sha256']==digest(archive)
        (out/'archive.json').write_text(json.dumps(identity,indent=2))
        if status['exit_code']==0:
            auditbase=root/'validation-audits'/trial
            commands=[
                [sys.executable,str(parent/'work/freeze_mlp_evidence.py'),str(archive),str(auditbase),identity['sha256']],
                [sys.executable,str(tools/'verify_ndarray_selective_native_checks.py'),str(auditbase),str(root/'ndarray-selective-native-checks-verification.json')]]
            for i,command in enumerate(commands):
                p=subprocess.run(command,capture_output=True,text=True)
                (out/('audit-'+str(i)+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0
            result=dict(validation_exit_code=0,archived=True,independent_audit_passed=True,archive=identity,full_goal_met=False)
            (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);code=0
        else:
            result=dict(validation_exit_code=status['exit_code'],archived=True,independent_audit_passed=False,archive=identity,full_goal_met=False)
            (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);code=2
        assert all(digest(Path(p))==sha for p,sha in identities.items())
    finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-begin)))
    sys.exit(code)
