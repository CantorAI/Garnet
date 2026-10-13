"""Sequential real legacy rejection/recovery and unchanged Qwen3.5 VL checks."""
import ast,hashlib,json,os,subprocess,sys,time
from pathlib import Path
r=Path('/Users/zsdx/garnet-metal-investigation');s=r/'source/test2026/metal'
output=r/'logs/finer-prefill-compatibility-20261013/jobs';output.mkdir(parents=True,exist_ok=False)
background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals())
runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3'
python=str(r/'comparators/vllm/venv/bin/python');statuses=[];records=[]
for batch in (1,2):
 for name in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
  vl=name=='Qwen3.5-0.8B';label=name+'-b'+str(batch)
  baseline=r/('logs/small-output-decode-vl-confirm-20261013/jobs' if vl else 'logs/equivalent-f32-linear-chunk-text-20261013/jobs')
  case=baseline/(label+'-case.json');folder=output/label;folder.mkdir()
  prefix='matched_frame_result ' if vl else 'matched_result '
  frozen=next(json.loads(v.split(' ',1)[1]) for v in (baseline/(label+'-garnet-0/output.log')).read_text().splitlines() if v.startswith(prefix))
  build=r/('build-small-output-decode' if vl else 'build-linear-chunk')
  expected='56bc06c693f3776c994d3f4078dd2e25093b575f93f6f09c4e2087a4917ee8d4' if vl else '26f2120c3570a28607969f855f6950d626372b388361325ac49096eb30d60671'
  assert hashlib.file_digest((build/'libgarnet.dylib').open('rb'),'sha256').hexdigest()==expected
  if vl:
   command=[python,str(s/'run_matched_frame_process.py'),runtime,str(build),str(r/'source'),str(r/'models'/name),str(case),'garnet',str(folder/'protocol')]
  else:
   command=[sys.executable,str(s/'benchmark_matched_process.py'),runtime,str(s/'run_dense_profile_guard.py'),str(build),str(r/'source'),str(r/'models'/name),str(case)]
  started=time.monotonic();waits=[]
  while True:
   try:guard();break
   except RuntimeError as e:
    waits.append(dict(elapsed_seconds=time.monotonic()-started,reason=str(e)));(folder/'wait.json').write_text(json.dumps(waits,indent=2))
    if time.monotonic()-started>900:raise
    time.sleep(5)
  (folder/'before-processes.txt').write_text(snapshot())
  with (folder/'output.log').open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
  statuses.append(dict(label=label,command=command,exit_code=result.returncode));(output/'statuses.json').write_text(json.dumps(statuses,indent=2))
  assert result.returncode==0,label
  guard();(folder/'after-processes.txt').write_text(snapshot())
  lines=(folder/'output.log').read_text().splitlines();values=[json.loads(v.split(' ',1)[1]) for v in lines if v.startswith(prefix)]
  assert len(values)==3,label
  if not vl:assert 'dense_profile_guard PASS' in lines
  for v in values:
   for key in ('token_ids','finished','prompt_token_ids'):assert v[key]==frozen[key],(label,key)
   assert all(v['finished'])
   if vl:assert v['prefill_chunk_plan']==([64,16] if batch==1 else [32,32,16])
  records.append(dict(label=label,complete_requests=len(values),exact_complete_outputs=True,legacy_rejections=not vl,qwen35_plan_unchanged=vl))
  (output/'summary.json').write_text(json.dumps(dict(scope='Compatibility only; no speed claim',records=records),indent=2))
  print(label,'passed',flush=True)
