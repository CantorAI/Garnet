
import pathlib,subprocess,json
r=pathlib.Path('/Users/zsdx/garnet-metal-investigation')
out=r/'logs/extended-dense-profile-checks-20261013/jobs'
out.mkdir(parents=True,exist_ok=False)
statuses=[]
for name in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'):
 guard=r/('guard-extended-dense-'+name+'.py')
 guard.write_text((r/'guard_extended_prefill.py').read_text().replace('extended-prefill-preflight-20261013','extended-dense-preflight-'+name+'-20261013'))
 subprocess.run(['/opt/homebrew/bin/python3',str(guard)],check=True)
 command=['/opt/homebrew/bin/python3',str(r/'source/test2026/metal/test_extended_dense_reference.py'),'/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3',str(r/'build-linear-chunk'),str(r/'source'),str(r/'models'/name),'bf16']
 with (out/(name+'.log')).open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
 statuses.append(dict(name=name,command=command,exit_code=result.returncode))
 (out/'statuses.json').write_text(json.dumps(statuses,indent=2))
 print(name,result.returncode,flush=True)
 if result.returncode:raise SystemExit(result.returncode)
