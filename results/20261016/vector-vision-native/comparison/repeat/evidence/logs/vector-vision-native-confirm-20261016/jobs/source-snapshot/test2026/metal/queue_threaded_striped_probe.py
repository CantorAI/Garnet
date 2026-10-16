"""Wait for the exact owned inference confirmation before any CPU timing."""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();prior=root/'logs/striped-frame-inference-confirm-20261015';pid=30402
assert json.loads((prior/'pid.json').read_text())['pid']==pid
if not (prior/'exit.json').exists():
    os.kill(pid,0);args=subprocess.check_output(['ps','-p',str(pid),'-o','args='],text=True)
    assert 'launch_confirm_striped_frame_inference.py' in args and str(prior) in args and '--worker' in args
if len(sys.argv)==3:
    out.mkdir(parents=True,exist_ok=False)
    with (out/'output.log').open('w') as stream:p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(out),'--worker'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    (out/'pid.json').write_text(json.dumps(dict(pid=p.pid)));print(json.dumps(dict(pid=p.pid,output=str(out))))
else:
    assert len(sys.argv)==4 and sys.argv[3]=='--worker';begin=time.monotonic();code=1
    try:
        while not (prior/'exit.json').exists():
            os.kill(pid,0)
            if time.monotonic()-begin>1800:raise TimeoutError('Owned inference confirmation dependency wait expired')
            time.sleep(5)
        status=json.loads((prior/'exit.json').read_text());assert status['exit_code']==0
        (out/'dependency.json').write_text(json.dumps(dict(pid=pid,status=status)))
        command=[str(root/'comparators/vllm/venv/bin/python'),str(root/'source/test2026/metal/run_threaded_striped_probe.py'),str(root),str(out/'jobs')]
        (out/'command.json').write_text(json.dumps(command));code=subprocess.call(command)
    finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-begin)))
    sys.exit(code)
