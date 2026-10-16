"""Sequential trained MLP measurement only after the exact primitive audit chain."""
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
    assert sys.argv[3]=='--worker';start=time.monotonic();code=1;state='waiting_exact_primitive_controller';commands=[]
    helpers=Path(__file__).parent
    def wait(folder,expected=None):
        pid=json.loads((folder/'pid.json').read_text())['pid']
        if expected is not None:assert pid==expected
        while not (folder/'exit.json').exists():
            live=subprocess.run(['ps','-p',str(pid),'-o','pid=,state=,etime=,comm='],capture_output=True,text=True)
            (out/'last-handle-observation.json').write_text(json.dumps(dict(pid=pid,folder=str(folder),observed_unix=time.time(),
                exit_code=live.returncode,observation=live.stdout),indent=2))
            if live.returncode!=0:
                time.sleep(1);assert (folder/'exit.json').exists(),'Exact handle missing without terminal record; do not restart'
            time.sleep(2)
        return json.loads((folder/'exit.json').read_text())
    def run(label,command):
        with (out/(label+'.log')).open('w') as stream:p=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        commands.append(dict(label=label,command=command,exit_code=p.returncode));(out/'commands.json').write_text(json.dumps(commands,indent=2))
        assert p.returncode==0,(label,p.returncode)
    try:
        dependency=root/'shape-selected-two-investigation-20261015/logs/shape-native-bf16-finalize-20261015'
        status=wait(dependency,53701)
        assert status['exit_code']==0 and status['state']=='native_and_bf16_primitive_audited'
        trial=root/'logs/bf16-feed-mlp-component-20261015';state='launching_trained_component'
        run('launch',[sys.executable,str(helpers/'launch_bf16_feed_mlp_component.py'),str(root),str(trial),'component'])
        state='waiting_exact_component';status=wait(trial)
        archive=root/'bf16-feed-mlp-component-20261015.tar.gz';state='freezing_terminal_component'
        run('archive',[sys.executable,str(helpers/'archive_bf16_feed_mlp_component.py'),str(root),str(archive)])
        assert status['exit_code']==0,'Terminal component failure preserved; no successful audit claimed'
        with archive.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        audit=root/'validation-audits/bf16-feed-mlp-component-20261015';state='auditing_component'
        run('extract',[sys.executable,str(root/'work/freeze_mlp_evidence.py'),str(archive),str(audit),sha])
        result=root/'bf16-feed-mlp-component-verification.json'
        run('audit',[sys.executable,str(helpers/'verify_bf16_feed_mlp_component.py'),str(audit),str(result)])
        shutil.copyfile(result,out/'component-verification.json');state='trained_component_audited_not_promoted';code=0
    finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,state=state,wall_seconds=time.monotonic()-start)))
    sys.exit(code)
