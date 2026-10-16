"""Actual VL QKV exact-half projection with integer rescaling; component only."""
import ast,hashlib,json,os,signal,statistics,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve();output.mkdir(parents=True,exist_ok=False)
scripts=root/'source/test2026/metal';background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((scripts/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals())
statuses=[]
def checked_guard(owned=None):
    guard()
    for line in snapshot().splitlines():
        fields=line.split()
        if fields and fields[-1].lower().endswith(('_gpu','_probe')) and int(fields[0])!=owned:raise RuntimeError('Other GPU probe PID '+fields[0])
def run(name,command):
    checked_guard();begin=time.monotonic();samples=[]
    with (output/(name+'.log')).open('w') as log:
        p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while p.poll() is None:
                checked_guard(p.pid);samples.append(dict(elapsed_seconds=time.monotonic()-begin,guard_passed=True))
                if time.monotonic()-begin>300:raise TimeoutError('Owned probe deadline')
                time.sleep(.5)
        except Exception:
            if p.poll() is None:
                assert os.getpgid(p.pid)==p.pid;os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=15)
            raise
    statuses.append(dict(name=name,command=command,exit_code=p.returncode,wall_seconds=time.monotonic()-begin))
    (output/'statuses.json').write_text(json.dumps(statuses,indent=2));(output/(name+'-monitor.json')).write_text(json.dumps(samples,indent=2));checked_guard()
    return p.returncode
source=scripts/'extended_half_bits_projection_probe.mm';binary=output/'extended_half_bits_projection_probe'
metadata=dict(scope=__doc__,source_sha256={str(p.relative_to(root/'source')):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in (source,Path(__file__),scripts/'experimental_extended_half_codec.h',scripts/'prepare_prefill_tile_probe.py',scripts/'compare_vllm_metal.py',scripts/'experimental_scale_bits.metal')})
(output/'metadata.json').write_text(json.dumps(metadata,indent=2))
assert run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders',str(source),'-o',str(binary)])==0
assets=output/'assets'
assert run('prepare',[sys.executable,str(scripts/'prepare_prefill_tile_probe.py'),str(root/'models/Qwen3.5-0.8B/model.safetensors'),str(assets)])==0
code=run('gpu',[str(binary),str(assets),str(scripts/'experimental_scale_bits.metal')])
records=[json.loads(line) for line in (output/'gpu.log').read_text().splitlines() if line.startswith('{')]
checks=[r for r in records if r['kind']=='check'];tiny=[r for r in records if r['kind']=='tiny_check'];results=[]
for check in checks:
    times=[r for r in records if r['kind']=='timing' and r['rows']==check['rows']];
    if not times:continue
    assert len(times)==8
    control=statistics.median(r['gpu_ms'] for r in times if not r['candidate']);candidate=statistics.median(r['gpu_ms'] for r in times if r['candidate'])
    results.append(dict(rows=check['rows'],control_gpu_ms=control,candidate_gpu_ms=candidate,component_speedup=control/candidate))
(output/'summary.json').write_text(json.dumps(dict(scope=__doc__,execution_exit_code=code,checks=checks,tiny_checks=tiny,diagnostics=[r for r in records if r['kind'].startswith('tiny_') and r['kind']!='tiny_check'],results=results,serving_changed=False),indent=2))
if code==0:assert len(checks)==8 and len(tiny)==8 and len(results)==8 and all(r['values']==r['rows']*14 for r in tiny) and all(r['candidate_worst_ratio']<=1 for r in records if r['kind']=='tiny_forward_diagnostic')
print(json.dumps(dict(execution_exit_code=code,checks=len(checks),tiny_checks=tiny,results=results)),flush=True)
raise SystemExit(code)
