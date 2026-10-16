"""Wait exact validation; archive/audit full28 before sequential native ABBA."""
from pathlib import Path
import json,os,subprocess,sys,tarfile,time

root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve()
checks=root/'logs/mlp-command-buffer-native-checks-20261015';expected_pid=5961
assert json.loads((checks/'pid.json').read_text())['pid']==expected_pid
if len(sys.argv)==3:
    out.mkdir(parents=True,exist_ok=False)
    with (out/'output.log').open('w') as stream:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(out),'--worker'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    (out/'pid.json').write_text(json.dumps(dict(pid=child.pid)))
    print(json.dumps(dict(pid=child.pid,output=str(out))))
else:
    assert sys.argv[3:] == ['--worker']
    start=time.monotonic();code=1;state='waiting_for_exact_validation';steps=[]
    tools=Path(__file__).resolve().parent
    def run(name,command):
        print('Starting',name,flush=True)
        with (out/(name+'.log')).open('w') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        steps.append(dict(name=name,command=command,exit_code=result.returncode))
        (out/'steps.json').write_bytes((json.dumps(steps,indent=2)+'\n').encode())
        assert result.returncode==0,name
        print('Completed',name,flush=True)
    try:
        while not (checks/'exit.json').exists():
            os.kill(expected_pid,0)
            args=subprocess.check_output(['ps','-p',str(expected_pid),'-o','args='],text=True)
            assert 'launch_command_buffer_native_checks.py' in args and str(checks) in args and '--worker' in args
            if time.monotonic()-start>7200:raise TimeoutError('Observation deadline; validation not restarted')
            time.sleep(5)
        status=json.loads((checks/'exit.json').read_text())
        archive=root/'command-buffer-native-checks-20261015.tar.gz'
        state='archiving_validation'
        run('archive-checks',[sys.executable,str(tools/'archive_command_buffer_native_checks.py'),str(root),str(archive)])
        assert status['exit_code']==0,'Validation failed; preserve terminal evidence and do not benchmark'
        audit=root/'validation-audits/command-buffer-native-checks-20261015';audit.mkdir(parents=True,exist_ok=False)
        with tarfile.open(archive) as tar:
            names=set()
            for member in tar:
                assert member.isfile() and member.name not in names;names.add(member.name)
                path=(audit/member.name).resolve();assert path.is_relative_to(audit)
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(tar.extractfile(member).read())
        state='auditing_original28'
        verification=root/'command-buffer-native-checks-verification.json'
        run('audit-checks',[sys.executable,str(tools/'verify_command_buffer_native_checks.py'),str(audit),str(verification),str(archive)])
        checked=json.loads(verification.read_text())
        assert checked['validation_stages']==28 and checked['dependent_graph_chain_cases']==64
        state='dispatching_native_profiles'
        native=root/'logs/mlp-command-buffer-native-20261015'
        run('launch-native',[sys.executable,str(tools/'launch_command_buffer_native_profiles.py'),str(root),str(native)])
        dispatched=json.loads((native/'pid.json').read_text());os.kill(dispatched['pid'],0)
        (out/'benchmark-dispatch.json').write_bytes((json.dumps(dict(**dispatched,output=str(native),checks_archive_sha256=checked['archive_sha256']),indent=2)+'\n').encode())
        state='native_benchmark_dispatched';code=0
    finally:
        (out/'exit.json').write_bytes((json.dumps(dict(exit_code=code,state=state,wall_seconds=time.monotonic()-start,
            scope='Validation archival/audit and native dispatch only; no model speed or vLLM gain claim'))+'\n').encode())
    sys.exit(code)
