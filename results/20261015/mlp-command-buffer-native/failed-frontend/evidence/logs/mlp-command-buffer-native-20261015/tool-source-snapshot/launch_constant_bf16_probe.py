"""Durable launcher for the exclusive real-weight indirect-command probe."""
import json,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve()
assert len(sys.argv)==3 or (len(sys.argv)==4 and sys.argv[3]=='--worker')
if len(sys.argv)==3:
 out.mkdir(parents=True,exist_ok=False)
 with (out/'output.log').open('w') as stream:
  process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(out),'--worker'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 (out/'pid.json').write_text(json.dumps(dict(pid=process.pid)))
 print(json.dumps(dict(pid=process.pid,output=str(out))))
else:
 begin=time.monotonic();code=1
 try:code=subprocess.call([sys.executable,str(root/'source/test2026/metal/run_constant_bf16_probe.py'),str(root),str(out/'jobs')])
 finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-begin)))
 sys.exit(code)
