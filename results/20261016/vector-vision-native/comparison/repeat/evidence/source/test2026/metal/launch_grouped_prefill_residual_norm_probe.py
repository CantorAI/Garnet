"""Durable guarded groups; discard whole interrupted groups and retry only those."""
import ast,hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();output=out
assert len(sys.argv)==3 or (len(sys.argv)==4 and sys.argv[3]=='--worker')
if len(sys.argv)==3:
 out.mkdir(parents=True,exist_ok=False)
 with (out/'output.log').open('w') as stream:
  p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(out),'--worker'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 (out/'pid.json').write_text(json.dumps(dict(pid=p.pid)));print(json.dumps(dict(pid=p.pid,output=str(out))))
else:
 begin=time.monotonic();code=1;groups=[];worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py');background_checks=[]
 nodes=[n for n in ast.parse((root/'source/test2026/metal/compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),'exclusive_guard','exec'),globals())
 try:
  for mode in ('regular','extra'):
   for index in range(6):
    for attempt in range(20):
     quiet=None;start=time.monotonic();waits=[]
     while True:
      now=time.monotonic()
      try:
       guard()
       if any(line.split()[-1].lower().endswith(('_gpu','_probe')) for line in snapshot().splitlines() if line.split()):raise RuntimeError('Another GPU probe is active')
      except RuntimeError as error:quiet=None;waits.append(dict(elapsed_seconds=now-start,reason=str(error)))
      else:
       if quiet is None:quiet=now
       if now-quiet>=5:break
      (out/'quiet-waits.json').write_text(json.dumps(waits,indent=2)+'\n')
      if now-start>900:raise TimeoutError('No five-second quiet interval')
      time.sleep(.5)
     name=mode+'-case'+str(index)+'-attempt'+str(attempt);folder=out/name;folder.mkdir()
     print('Starting exclusive group '+name,flush=True);command=[sys.executable,str(root/'source/test2026/metal/run_grouped_prefill_residual_norm_probe.py'),str(root),str(folder/'jobs'),str(index),mode];start=time.monotonic()
     with (folder/'output.log').open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=1200)
     (folder/'exit.json').write_text(json.dumps(dict(exit_code=result.returncode,wall_seconds=time.monotonic()-start,command=command)))
     if result.returncode:
      log=(folder/'output.log').read_text()
      if 'Another inference runtime is active' not in log and 'Other GPU probe PID' not in log:raise RuntimeError('Group failed for non-contention reason: '+name)
      print('Discarded interrupted group '+name,flush=True);continue
     groups.append(dict(mode=mode,index=index,evidence=name));(out/'summary.json').write_text(json.dumps(dict(groups=groups,all_completed=len(groups)==12),indent=2)+'\n')
     print('Completed clean group '+name,flush=True);break
    else:raise RuntimeError('Repeated contention exhausted this group')
  assert len(groups)==12;code=0
 finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-begin)))
 sys.exit(code)
