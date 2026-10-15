"""Isolated three-half activation representation; no serving/precision speed claim."""
import ast,hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
r=Path(sys.argv[1]);output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=False)
s=r/'source/test2026/metal';statuses=[];background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals());strict_guard=guard
def guard(label):
    strict_guard()
    for line in snapshot().splitlines():
        f=line.split()
        if f and f[-1].lower().endswith(('/split_half_prefill_probe','/lossless_block_gpu_probe','/lossless_gpu_probe','/prefill_tile_gpu','/parallel_delta_gpu')):
            raise RuntimeError('Other GPU probe PID '+f[0])
    (output/(label+'-processes.txt')).write_text(snapshot())
def run(name,command):
    with (output/(name+'.log')).open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
    statuses.append(dict(name=name,command=command,exit_code=result.returncode))
    (output/'statuses.json').write_text(json.dumps(statuses,indent=2));assert result.returncode==0,(name,result.returncode)
metadata=dict(scope=__doc__,source_sha256={p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in [Path(__file__),s/'split_half_prefill_probe.mm',s/'split_half_prefill_probe.metal',s/'prepare_prefill_tile_probe.py',r/'source/src/backends/metal/exact_half_codec.h']})
(output/'metadata.json').write_text(json.dumps(metadata,indent=2));binary=output/'split_half_prefill_probe'
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders',str(s/'split_half_prefill_probe.mm'),'-o',str(binary)])
checks=[];rows=[]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    guard(model+'-before');assets=output/(model+'-assets')
    run(model+'-prepare',[sys.executable,str(s/'prepare_prefill_tile_probe.py'),str(r/'models'/model/'model.safetensors'),str(assets)])
    guard(model+'-admission-before');run(model+'-admission',[str(binary),str(s/'split_half_prefill_probe.metal'),str(assets),'admission']);
    guard(model+'-gpu-before');run(model+'-gpu',[str(binary),str(s/'split_half_prefill_probe.metal'),str(assets)]);guard(model+'-after')
    info=json.loads((assets/'manifest.json').read_text())['cases'];records=[json.loads(v) for v in (output/(model+'-gpu.log')).read_text().splitlines() if v.startswith('{')]
    quality=[v for v in records if v['kind']=='check'];assert len(quality)==len(info)*4
    checks.extend(dict(model=model,**v) for v in quality)
    for case in info:
        for count in (24,32,64,80):
            times=[v for v in records if v['kind']=='timing' and v['name']==case['name'] and v['rows']==count];assert len(times)==8
            control=statistics.median(v['gpu_ms'] for v in times if not v['candidate']);candidate=statistics.median(v['gpu_ms'] for v in times if v['candidate'])
            rows.append(dict(model=model,**case,rows=count,control_ms=control,candidate_ms=candidate,ratio=control/candidate))
    (output/'summary.json').write_text(json.dumps(dict(scope=metadata['scope'],checks=checks,results=rows),indent=2))
    print('Completed component',model,'checks',len(quality),flush=True)
