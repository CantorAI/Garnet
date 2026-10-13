"""Real longer-prompt output comparison: opt-in chunk 48 against chunk 64."""
import ast,hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
r=Path('/Users/zsdx/garnet-metal-investigation');s=r/'source/test2026/metal'
output=r/'logs/finer-prefill-48-requests-20261013/jobs';output.mkdir(parents=True,exist_ok=False)
background_checks=[];worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals())
runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3';build=r/'build-small-output-decode'
assert hashlib.file_digest((build/'libgarnet.dylib').open('rb'),'sha256').hexdigest()=='56bc06c693f3776c994d3f4078dd2e25093b575f93f6f09c4e2087a4917ee8d4'
statuses=[];groups=[]
prompts={'Qwen3-0.6B':'What is the capital of France? Answer briefly with only the city name. Read this instruction carefully and give a short direct answer without any extra explanation.',
 'Qwen2.5-Coder-0.5B-Instruct':'Write a Python function add(a, b) that returns their sum. Output only code. Use the exact function name add and these two parameter names. Do not add any explanation.'}
for batch in (1,2):
 for name,prompt in prompts.items():
  values={}
  for backend in ('control','candidate'):
   folder=output/(name+'-b'+str(batch)+'-'+backend);folder.mkdir()
   case=dict(prompt=prompt,batch=batch,max_tokens=128,capacity=256,chunk=64,requests=3,prefill_projection='shared',prefill_finer_chunks=backend=='candidate')
   casefile=folder/'case.json';casefile.write_text(json.dumps(case,indent=2))
   started=time.monotonic();waits=[]
   while True:
    try:guard();break
    except RuntimeError as e:
     waits.append(dict(elapsed_seconds=time.monotonic()-started,reason=str(e)));(folder/'wait.json').write_text(json.dumps(waits,indent=2))
     if time.monotonic()-started>900:raise
     time.sleep(5)
   (folder/'before-processes.txt').write_text(snapshot())
   command=[sys.executable,str(s/'benchmark_matched_process.py'),runtime,str(s/'run_matched_garnet.py'),str(build),str(r/'source'),str(r/'models'/name),str(casefile)]
   with (folder/'output.log').open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
   statuses.append(dict(folder=folder.name,command=command,exit_code=result.returncode));(output/'statuses.json').write_text(json.dumps(statuses,indent=2))
   assert result.returncode==0,folder
   guard();(folder/'after-processes.txt').write_text(snapshot())
   rows=[json.loads(v.split(' ',1)[1]) for v in (folder/'output.log').read_text().splitlines() if v.startswith('matched_result ')]
   assert len(rows)==3
   for v in rows:
    assert 32<len(v['prompt_token_ids'])<=48,(name,len(v['prompt_token_ids']))
    assert v['prefill_chunk_plan']==([48] if backend=='candidate' else [64])
    assert all(v['finished']),name
   values[backend]=rows
  for a,b in zip(values['control'],values['candidate']):
   for key in ('token_ids','finished','prompt_token_ids'):assert a[key]==b[key],(name,batch,key)
  group=dict(model=name,batch=batch,prompt_tokens=len(values['control'][0]['prompt_token_ids']),exact_complete_outputs=True,records=values)
  groups.append(group);(output/'summary.json').write_text(json.dumps(dict(scope='Longer prompt chunk-48 compatibility; different from matched comparator workload; no target claim',groups=groups),indent=2))
  print(name,'batch',batch,'passed with',group['prompt_tokens'],'prompt tokens',flush=True)
