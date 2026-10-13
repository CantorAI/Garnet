import ast,hashlib,json,os,subprocess,time
from pathlib import Path
r=Path('/Users/zsdx/garnet-metal-investigation');s=r/'source/test2026/metal';output=r/'logs/linear-conv-chunk-checks-20261013/jobs';output.mkdir(parents=True,exist_ok=False)
background_checks=[];worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals())
runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3';py='/opt/homebrew/bin/python3';build=str(r/'build-linear-conv-chunk');statuses=[]
commands=[('ctest',['/opt/homebrew/bin/ctest','--test-dir',build,'--output-on-failure'])]
for label,script in [('serving','test_decoder_model.py'),('mixed-image','test_image_pipeline.py')]:commands.append((label,[py,str(s/script),runtime,build,str(r/'source')]))
commands.append(('linear-reference',[py,str(s/'test_direct_write_linear_reference.py'),runtime,build,str(r/'source'),str(r/'models/Qwen3.5-0.8B'),'bf16']))
for label,command in commands:
 guard();print('Starting',label,flush=True)
 with (output/(label+'.log')).open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
 statuses.append(dict(label=label,command=command,exit_code=result.returncode));(output/'statuses.json').write_text(json.dumps(statuses,indent=2))
 assert result.returncode==0,(label,result.returncode)
 guard();print('Passed',label,flush=True)
