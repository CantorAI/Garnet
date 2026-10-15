"""Durable dependency gate; verify the live owned check process before waiting."""
import json,shlex,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();prior=root/'logs/constant-head-checks-20261014'
assert len(sys.argv)==3 or (len(sys.argv)==4 and sys.argv[3]=='--worker')
if len(sys.argv)==3:
 out.mkdir(parents=True,exist_ok=False)
 with (out/'output.log').open('w') as stream:
  process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(out),'--worker'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 (out/'pid.json').write_text(json.dumps(dict(pid=process.pid)))
 print(json.dumps(dict(pid=process.pid,output=str(out))))
else:
 begin=time.monotonic();code=1
 try:
  while not (prior/'exit.json').exists():
   pid=json.loads((prior/'pid.json').read_text())['pid'];assert pid==10827
   probe=subprocess.run(['/bin/ps','-p',str(pid),'-o','args='],capture_output=True,text=True)
   if (prior/'exit.json').exists():break
   assert probe.returncode==0 and shlex.split(probe.stdout.strip())[1:]==[str(root/'source/test2026/metal/launch_constant_head_checks.py'),str(root),str(prior),'--worker'],'Owned check handle missing or changed'
   (out/'dependency-wait.json').write_text(json.dumps(dict(prior_pid=pid,elapsed_seconds=time.monotonic()-begin,verified_live=True)))
   if time.monotonic()-begin>1800:raise TimeoutError('Owned numerical checks exceeded dependency deadline')
   time.sleep(5)
  assert json.loads((prior/'exit.json').read_text())['exit_code']==0
  print('All numerical checks passed; starting sequential complete-output head comparisons',flush=True)
  code=subprocess.call([sys.executable,str(root/'source/test2026/metal/run_constant_head_profiles.py'),str(root),str(out/'jobs')])
 finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-begin)))
 sys.exit(code)
