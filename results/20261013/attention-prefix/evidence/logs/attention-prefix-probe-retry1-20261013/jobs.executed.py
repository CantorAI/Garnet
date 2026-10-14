"""Sequential component experiment; no serving performance claim."""
import ast, hashlib, json, os, signal, statistics, subprocess, sys, time
from pathlib import Path

root=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve()
output.mkdir(parents=True,exist_ok=False);scripts=root/'source/test2026/metal'
background_checks=[];statuses=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((scripts/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals())
strict_guard=guard
def check(owned=None):
    strict_guard()
    for line in snapshot().splitlines():
        fields=line.split()
        if fields and fields[-1].lower().endswith(('_gpu','_probe')) and int(fields[0])!=owned:
            raise RuntimeError('Another GPU probe PID '+fields[0])
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def run(name,command):
    check();begin=time.monotonic();samples=[]
    with (output/(name+'.log')).open('w') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                check(process.pid);samples.append(dict(elapsed_seconds=time.monotonic()-begin,guard_passed=True))
                if time.monotonic()-begin>600:raise TimeoutError('Owned component deadline exceeded')
                time.sleep(.5)
            code=process.returncode
        except BaseException:
            if process.poll() is None:
                assert os.getpgid(process.pid)==process.pid;os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=15)
            raise
    (output/(name+'-monitor.json')).write_text(json.dumps(samples,indent=2)+'\n')
    statuses.append(dict(name=name,command=command,exit_code=code));(output/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n')
    if code:raise RuntimeError(name+' failed; log retained')

native=root/'source/src/backends/metal/metal_context.mm'
paths=[native,Path(__file__),scripts/'attention_prefix_gpu.mm',scripts/'make_attention_prefix_shader.py',scripts/'compare_vllm_metal.py']
(output/'metadata.json').write_text(json.dumps(dict(scope='FP32 chunk attention verified-prefix component only; synthetic inputs, general-scan fallback, no full model claim',source_sha256={str(p.relative_to(root/'source')):digest(p) for p in paths},hardware={k:subprocess.check_output(['/usr/sbin/sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize')}),indent=2)+'\n')
(output/'before-processes.txt').write_text(snapshot());check()
shader=output/'combined.metal';binary=output/'attention_prefix_gpu'
run('shader',[sys.executable,str(scripts/'make_attention_prefix_shader.py'),str(native),str(shader)])
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders',str(scripts/'attention_prefix_gpu.mm'),'-o',str(binary)])
check();time.sleep(5);check();run('gpu',[str(binary),str(shader)])
rows=[json.loads(line) for line in (output/'gpu.log').read_text().splitlines() if line.startswith('{')]
checks=[r for r in rows if r['kind']=='check'];assert len(checks)==18
results=[]
for item in checks:
    times=[r for r in rows if r['kind']=='timing' and all(r[k]==item[k] for k in ('batch','chunk','heads','kv_heads','dim'))];assert len(times)==8
    control=statistics.median(r['gpu_ms_median'] for r in times if r['policy']=='control')
    candidate=statistics.median(r['gpu_ms_median'] for r in times if r['policy']=='prefix')
    results.append(dict(**item,control_gpu_ms=control,prefix_gpu_ms=candidate,component_speedup=control/candidate))
(output/'summary.json').write_text(json.dumps(dict(checks=checks,results=results),indent=2)+'\n')
(output/'after-processes.txt').write_text(snapshot());check()
print(json.dumps(dict(checks=len(checks),timings=len(rows)-len(checks),worst_fp64_ratio=max(r['fp64_worst_ratio'] for r in checks),serving_shape=[r for r in results if r['heads']==8 and r['dim']==256])))
