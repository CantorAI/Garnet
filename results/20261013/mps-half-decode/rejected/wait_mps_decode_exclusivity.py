import json,subprocess,time,hashlib,os,ast
from pathlib import Path
source=Path('/Users/zsdx/garnet-metal-investigation/source');output=source.parent/'logs/mps-half-decode-exclusivity-wait-20261013/jobs'
import ast,hashlib,os,time
background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
output.mkdir(parents=True,exist_ok=True)
nodes=[n for n in ast.parse((source/'test2026/metal/compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals())

start=time.monotonic();attempts=[]
while True:
 try:guard();passed=True
 except RuntimeError as e:passed=False;reason=str(e)
 attempts.append(dict(elapsed_seconds=time.monotonic()-start,passed=passed))
 (output/'attempts.json').write_text(json.dumps(attempts,indent=2)+'\n')
 if passed:
  (output/'final-processes.txt').write_text(snapshot());print('Verified inference exclusivity',flush=True);break
 if time.monotonic()-start>900:raise RuntimeError('Other unverified process still active after bounded exclusivity wait')
 time.sleep(5)
