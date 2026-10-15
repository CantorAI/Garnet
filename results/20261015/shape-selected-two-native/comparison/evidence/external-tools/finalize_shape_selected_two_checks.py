"""Wait for exact validation PID, freeze/audit it, then dispatch native ABBA."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,time
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve()
if len(sys.argv)==3:
    out.mkdir(parents=True,exist_ok=False)
    with (out/'output.log').open('w') as stream:
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(out),'--worker'],
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    (out/'pid.json').write_text(json.dumps(dict(pid=p.pid)));print(json.dumps(dict(pid=p.pid,output=str(out))))
else:
    assert sys.argv[3]=='--worker'
    start=time.monotonic();code=1;state='waiting_for_exact_validation';commands=[]
    helpers=Path(__file__).parent;parent=Path(json.loads((root/'preparation.json').read_text())['parent'])
    trial=root/'logs/shape-selected-two-checks-20261015';handle=json.loads((trial/'pid.json').read_text())
    assert handle['pid']==42435
    def run(label,command):
        with (out/(label+'.log')).open('w') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        commands.append(dict(label=label,command=command,exit_code=result.returncode))
        (out/'commands.json').write_text(json.dumps(commands,indent=2))
        assert result.returncode==0,(label,result.returncode)
    try:
        while not (trial/'exit.json').exists():
            live=subprocess.run(['ps','-p',str(handle['pid']),'-o','pid=,state=,etime=,comm='],capture_output=True,text=True)
            (out/'last-handle-observation.json').write_text(json.dumps(dict(pid=handle['pid'],exit_code=live.returncode,observation=live.stdout,observed_unix=time.time()),indent=2))
            if live.returncode!=0:
                time.sleep(1)
                assert (trial/'exit.json').exists(),'Known validation handle missing without terminal record; do not restart'
            time.sleep(2)
        status=json.loads((trial/'exit.json').read_text());state='archiving_terminal_validation'
        archive=root/'shape-selected-two-checks-20261015.tar.gz'
        run('archive',[sys.executable,str(helpers/'archive_shape_selected_two_checks.py'),str(root),str(archive)])
        with archive.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        (out/'validation-terminal.json').write_text(json.dumps(dict(pid=handle['pid'],status=status,archive_sha256=sha),indent=2))
        assert status['exit_code']==0,'Terminal failure archived; no native benchmark admitted'
        audit=root/'validation-audits/shape-selected-two-checks-20261015'
        run('extract',[sys.executable,str(parent/'work/freeze_mlp_evidence.py'),str(archive),str(audit),sha])
        result=root/'shape-selected-two-checks-verification.json'
        run('audit',[sys.executable,str(helpers/'verify_shape_selected_two_checks.py'),str(audit),str(result),str(archive)])
        shutil.copyfile(result,out/'checks-verification.json')
        state='dispatching_native_abba'
        run('launch-native',[sys.executable,str(helpers/'launch_shape_selected_two_profiles.py'),str(root),str(root/'logs/shape-selected-two-native-20261015'),'profiles'])
        state='native_benchmark_dispatched';code=0
    finally:
        (out/'exit.json').write_text(json.dumps(dict(exit_code=code,state=state,wall_seconds=time.monotonic()-start)))
    sys.exit(code)
