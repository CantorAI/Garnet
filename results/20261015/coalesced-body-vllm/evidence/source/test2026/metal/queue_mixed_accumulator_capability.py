"""Wait for exact live sparse profiler before any other GPU work."""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();dependency=root/'logs/counter-trace-sparse-profiles-20261015'
assert len(sys.argv)==3 or (len(sys.argv)==4 and sys.argv[3]=='--worker')
assert json.loads((dependency/'pid.json').read_text())['pid']==62176
if not (dependency/'exit.json').exists():
    os.kill(62176,0);args=subprocess.check_output(['ps','-p','62176','-o','args='],text=True)
    assert 'launch_sparse_counter_trace_profiles.py' in args and str(dependency) in args and '--worker' in args
if len(sys.argv)==3:
    out.mkdir(parents=True,exist_ok=False)
    with (out/'output.log').open('w') as log:p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(out),'--worker'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (out/'pid.json').write_text(json.dumps(dict(pid=p.pid)));print(json.dumps(dict(pid=p.pid,output=str(out))))
else:
    start=time.monotonic();code=1
    try:
        while not (dependency/'exit.json').exists():
            os.kill(62176,0)
            if time.monotonic()-start>1800:raise TimeoutError('Sparse profiling dependency wait expired')
            time.sleep(5)
        status=json.loads((dependency/'exit.json').read_text());assert status['exit_code']==0
        (out/'dependency.json').write_text(json.dumps(dict(pid=62176,status=status)))
        code=subprocess.call([sys.executable,str(root/'source/test2026/metal/run_mixed_accumulator_capability.py'),str(root),str(out/'jobs')])
    finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-start)))
    sys.exit(code)
