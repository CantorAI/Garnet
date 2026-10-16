"""Wait for a verified quiet interval before an exclusive packed prefill probe."""
import ast,hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();output=out;mode=sys.argv[3]
assert mode in ('regular','extra') and (len(sys.argv)==4 or sys.argv[4]=='--worker')
if len(sys.argv)==4:
 out.mkdir(parents=True,exist_ok=False)
 with (out/'output.log').open('w') as stream:
  p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(out),mode,'--worker'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 (out/'pid.json').write_text(json.dumps(dict(pid=p.pid)));print(json.dumps(dict(pid=p.pid,output=str(out))))
else:
 start=time.monotonic();code=1;waits=[];quiet=None
 try:
  worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py');background_checks=[]
  nodes=[n for n in ast.parse((root/'source/test2026/metal/compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
  exec(compile(ast.Module(body=nodes,type_ignores=[]),'exclusive_guard','exec'),globals())
  while True:
   now=time.monotonic()
   try:guard()
   except RuntimeError as error:quiet=None;waits.append(dict(elapsed_seconds=now-start,reason=str(error)))
   else:
    if quiet is None:quiet=now
    if now-quiet>=5:break
   (out/'quiet-waits.json').write_text(json.dumps(waits,indent=2)+'\n')
   if now-start>900:raise TimeoutError('No five-second quiet inference interval')
   time.sleep(.5)
  runner='run_prefill_residual_norm_probe.py' if mode=='regular' else 'run_extra_prefill_residual_norm_probe.py'
  print('Quiet interval verified; starting exclusive '+mode+' probe',flush=True)
  code=subprocess.call([sys.executable,str(root/'source/test2026/metal'/runner),str(root),str(out/'jobs')])
 finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-start)))
 sys.exit(code)
