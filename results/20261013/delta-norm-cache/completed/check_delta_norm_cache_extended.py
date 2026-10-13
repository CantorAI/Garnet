import ast,hashlib,json,os,subprocess,time
from pathlib import Path
r=Path('/Users/zsdx/garnet-metal-investigation');output=r/'logs/delta-norm-cache-extended-numerical-20261013/jobs';output.mkdir(parents=True,exist_ok=False);background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((r/'source/test2026/metal/compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals())
started=time.monotonic();waits=[]
while True:
 try:guard();break
 except RuntimeError as e:
  waits.append(dict(elapsed_seconds=time.monotonic()-started,reason=str(e)));(output/'wait.json').write_text(json.dumps(waits,indent=2))
  if time.monotonic()-started>900:raise
  time.sleep(5)
command=['/opt/homebrew/bin/python3',str(r/'source/test2026/metal/test_qwen35_extended_prefill_reference.py'),'/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3',str(r/'build-delta-norm-cache'),str(r/'source'),str(r/'models/Qwen3.5-0.8B'),str(output/'failures')]
with (output/'reference.log').open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=2400)
(output/'status.json').write_text(json.dumps(dict(command=command,exit_code=result.returncode),indent=2))
assert result.returncode==0,result.returncode
lines=(output/'reference.log').read_text().splitlines();rows=[v for v in lines if v.startswith('chunk reference ')]
assert len(rows)==36,len(rows)
print('Passed 36 extended real-weight reference comparisons',flush=True)
