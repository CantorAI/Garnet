"""Actual independent trained references for opt-in 22/26-token dense profiles."""
import ast,hashlib,json,os,subprocess,sys,time
from pathlib import Path
r=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=False);s=r/'source/test2026/metal';statuses=[];background_checks=[]
output=out
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'guard','exec'),globals())
runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3';build=r/'build-resident-kv-inplace'
metadata=dict(library_sha256=hashlib.file_digest((build/'libgarnet.dylib').open('rb'),'sha256').hexdigest(),source_sha256={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in [Path(__file__),s/'test_exact_dense_tail_reference.py',s/'run_dense_reference.py',s/'run_matched_garnet.py',r/'source/xModel/qwen/metal_dense/decoder.py']})
assert metadata['library_sha256']=='73c10e5dc62eb97607e9ded9a1df8bb6702d87375620bd5dc160d8b871f78ce2';(out/'metadata.json').write_text(json.dumps(metadata,indent=2))
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'):
    guard();(out/(model+'-before-processes.txt')).write_text(snapshot())
    command=[sys.executable,str(s/'test_exact_dense_tail_reference.py'),runtime,str(build),str(r/'source'),str(r/'models'/model),'bf16']
    print('Starting independent exact-tail references',model,flush=True)
    with (out/(model+'.log')).open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
    statuses.append(dict(model=model,command=command,exit_code=result.returncode));(out/'statuses.json').write_text(json.dumps(statuses,indent=2));assert result.returncode==0,model
    guard();(out/(model+'-after-processes.txt')).write_text(snapshot());print('Completed exact-tail references',model,flush=True)
assert len(statuses)==2
