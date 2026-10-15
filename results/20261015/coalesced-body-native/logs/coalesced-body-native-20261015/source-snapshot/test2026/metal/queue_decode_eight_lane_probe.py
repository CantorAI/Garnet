"""Verify owned combined handle; archive its terminal evidence before one probe."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve()
assert len(sys.argv)==3 or (len(sys.argv)==4 and sys.argv[3]=='--worker')
dependency=root/'logs/combined-prefill-reset-20261015'
assert json.loads((dependency/'pid.json').read_text())['pid']==27124
if not (dependency/'exit.json').exists():
    os.kill(27124,0)
    # Inspect exact owned argv internally; never print unrelated process argv.
    args=subprocess.check_output(['ps','-p','27124','-o','args='],text=True).strip()
    assert 'launch_combined_prefill_reset.py' in args and str(dependency) in args and '--worker' in args
if len(sys.argv)==3:
    out.mkdir(parents=True,exist_ok=False)
    with (out/'output.log').open('w') as stream:
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(out),'--worker'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    (out/'pid.json').write_text(json.dumps(dict(pid=p.pid)))
    print(json.dumps(dict(pid=p.pid,output=str(out))))
else:
    begin=time.monotonic();code=1
    try:
        while not (dependency/'exit.json').exists():
            os.kill(27124,0)
            if time.monotonic()-begin>7200:raise TimeoutError('Combined dependency wait expired')
            time.sleep(5)
        assert json.loads((dependency/'exit.json').read_text())['exit_code']==0
        archive=root/'combined-prefill-reset-evidence.tar.gz'
        command=[sys.executable,str(root/'source/test2026/metal/archive_combined_prefill_reset.py'),str(root),str(archive)]
        result=subprocess.run(command,capture_output=True,text=True)
        (out/'dependency-archive.log').write_text(result.stdout+result.stderr)
        assert result.returncode==0
        (out/'dependency.json').write_text(json.dumps(dict(pid=27124,status=json.loads((dependency/'exit.json').read_text()),archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),command=command),indent=2))
        code=subprocess.call([sys.executable,str(root/'source/test2026/metal/run_decode_eight_lane_probe.py'),str(root),str(out/'jobs')])
    finally:
        (out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-begin)))
    sys.exit(code)
