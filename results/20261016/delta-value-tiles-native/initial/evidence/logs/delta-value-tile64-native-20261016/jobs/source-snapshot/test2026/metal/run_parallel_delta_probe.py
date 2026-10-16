"""One isolated GPU component prototype; preserve numerical failures/timings."""
import ast,hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
r=Path(sys.argv[1]);output=Path(sys.argv[2]);output.mkdir(parents=True,exist_ok=False);s=r/'source/test2026/metal';background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'guard','exec'),globals());statuses=[]
def run(name,command):
 with (output/(name+'.log')).open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
 statuses.append(dict(name=name,command=command,exit_code=result.returncode));(output/'statuses.json').write_text(json.dumps(statuses,indent=2));assert result.returncode==0,(name,result.returncode)
native=(r/'source/src/backends/metal/metal_context.mm').read_text();a=native.index('struct DeltaChunkParams');b=native.index('struct LayoutParams',a)
shader=output/'combined.metal';shader.write_text((s/'parallel_delta_probe.metal').read_text()+native[a:b])
metadata=dict(scope='FP32 GPU core component only; no full model or vLLM speed claim',source_sha256={str(p.relative_to(r)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in [s/'parallel_delta_probe.metal',s/'parallel_delta_gpu.mm',s/'prepare_parallel_delta_probe.py',Path(__file__),r/'source/src/backends/metal/metal_context.mm']})
(output/'metadata.json').write_text(json.dumps(metadata,indent=2))
binary=output/'parallel_delta_gpu';run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders',str(s/'parallel_delta_gpu.mm'),'-o',str(binary)])
run('prepare',[sys.executable,str(s/'prepare_parallel_delta_probe.py'),str(r/'models/Qwen3.5-0.8B'),str(output/'assets')])
guard();(output/'before-processes.txt').write_text(snapshot());run('gpu',[str(binary),str(shader),str(output/'assets')]);guard();(output/'after-processes.txt').write_text(snapshot())
lines=[json.loads(v) for v in (output/'gpu.log').read_text().splitlines() if v.startswith('{')];checks=[v for v in lines if v['kind']=='check'];cases=json.loads((output/'assets/manifest.json').read_text())['cases'];assert len(checks)==len(cases)
rows=[]
for case in cases:
 timings=[v for v in lines if v['kind']=='timing' and v['name']==case['name']];assert len(timings)==8
 control=statistics.median(v['gpu_ms'] for v in timings if not v['candidate']);candidate=statistics.median(v['gpu_ms'] for v in timings if v['candidate'])
 rows.append(dict(**case,control_gpu_ms=control,candidate_gpu_ms=candidate,ratio=control/candidate))
(output/'summary.json').write_text(json.dumps(dict(scope=metadata['scope'],checks=checks,results=rows),indent=2));print('GPU component passed',len(checks),flush=True)
