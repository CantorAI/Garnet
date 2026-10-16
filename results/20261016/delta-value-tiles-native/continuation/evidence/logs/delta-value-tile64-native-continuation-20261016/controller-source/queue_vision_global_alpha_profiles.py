"""Verify the exact live prefill validation handle before waiting for its result."""
import json,shlex,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();prior=root/'logs/vision-global-alpha-checks-20261014'
assert len(sys.argv)==3 or (len(sys.argv)==4 and sys.argv[3]=='--worker')
if len(sys.argv)==3:
 out.mkdir(parents=True,exist_ok=False)
 with (out/'output.log').open('w') as stream:
  p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(out),'--worker'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 (out/'pid.json').write_text(json.dumps(dict(pid=p.pid)));print(json.dumps(dict(pid=p.pid,output=str(out))))
else:
 begin=time.monotonic();code=1
 try:
  while not (prior/'exit.json').exists():
   pid=json.loads((prior/'pid.json').read_text())['pid'];assert pid==13040
   probe=subprocess.run(['/bin/ps','-p',str(pid),'-o','args='],capture_output=True,text=True)
   if (prior/'exit.json').exists():break
   assert probe.returncode==0 and shlex.split(probe.stdout.strip())[1:]==[str(root/'source/test2026/metal/launch_vision_global_alpha_checks.py'),str(root),str(prior),'checks','--worker'],'Owned validation handle missing or changed'
   (out/'dependency-wait.json').write_text(json.dumps(dict(prior_pid=pid,elapsed_seconds=time.monotonic()-begin,verified_live=True)))
   if time.monotonic()-begin>2400:raise TimeoutError('Validation exceeded dependency deadline')
   time.sleep(5)
  assert json.loads((prior/'exit.json').read_text())['exit_code']==0
  print('All validation stages passed; starting six sequential complete-output profiles',flush=True)
  code=subprocess.call([sys.executable,str(root/'source/test2026/metal/run_vision_global_alpha_profiles.py'),str(root),str(out/'jobs')])
 finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-begin)))
 sys.exit(code)
