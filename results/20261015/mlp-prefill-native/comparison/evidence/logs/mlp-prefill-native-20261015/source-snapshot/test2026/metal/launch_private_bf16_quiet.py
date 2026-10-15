"""Wait for a verified quiet interval, then run one exclusive private-weight gate."""
import ast,hashlib,json,os,signal,statistics,subprocess,sys,time,shutil
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
 try:
  output=out/'preflight';output.mkdir(exist_ok=False);background_checks=[]
  worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
  text=(root/'source/test2026/metal/compare_vllm_metal.py').read_text()
  nodes=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
  exec(compile(ast.Module(body=nodes,type_ignores=[]),'exclusive_preflight','exec'),globals())
  quiet_since=None
  while True:
   now=time.monotonic()
   try:
    guard()
    for line in snapshot().splitlines():
     fields=line.split()
     if fields and fields[-1].lower().endswith(('_gpu','_probe')):raise RuntimeError('Other GPU probe')
   except RuntimeError:
    quiet_since=None
   else:
    if quiet_since is None:quiet_since=now
    if now-quiet_since>=5:break
   (output/'wait.json').write_text(json.dumps(dict(elapsed_seconds=now-begin,quiet_seconds=0 if quiet_since is None else now-quiet_since,benchmark_started=False)))
   if now-begin>2700:raise TimeoutError('No verified quiet interval within 45 minutes')
   time.sleep(2)
  (output/'wait.json').write_text(json.dumps(dict(elapsed_seconds=time.monotonic()-begin,quiet_seconds=5,benchmark_started=True)))
  print('Five-second quiet guard interval passed; starting exclusive private-weight gate',flush=True)
  code=subprocess.call([sys.executable,str(root/'source/test2026/metal/run_private_bf16_probe.py'),str(root),str(out/'jobs')])
 finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-begin)))
 sys.exit(code)
