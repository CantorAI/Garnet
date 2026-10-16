"""Durable exclusive reset-component launcher; never change selected libraries."""
import json,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve()
assert len(sys.argv)==3 or (len(sys.argv)==4 and sys.argv[3]=='--worker')
if len(sys.argv)==3:
 output.mkdir(parents=True,exist_ok=False)
 with (output/'output.log').open('w') as log:p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(output),'--worker'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (output/'pid.json').write_bytes((json.dumps(dict(pid=p.pid))+'\n').encode());print(json.dumps(dict(pid=p.pid,output=str(output))))
else:
 begin=time.monotonic();code=1
 try:code=subprocess.call([sys.executable,str(root/'source/test2026/metal/run_dirty_prefix_reset_probe.py'),str(root),str(output/'jobs')])
 finally:(output/'exit.json').write_bytes((json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-begin))+'\n').encode())
 sys.exit(code)
