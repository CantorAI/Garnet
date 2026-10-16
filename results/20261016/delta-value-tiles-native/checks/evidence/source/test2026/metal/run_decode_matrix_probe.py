"""Exclusive real-weight sub-SIMD decode component; no full-model speed claim."""
import ast,hashlib,json,os,signal,statistics,subprocess,sys,time,shutil
from pathlib import Path
r=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve();output.mkdir(parents=True,exist_ok=False)
s=r/'source/test2026/metal';statuses=[];background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals());strict_guard=guard
def guard(owned=None):
    if shutil.disk_usage(r).free<2*2**30:raise OSError('Disk free space below 2 GiB')
    strict_guard()
    for line in snapshot().splitlines():
        f=line.split()
        if f and f[-1].lower().endswith(('_gpu','_probe')) and int(f[0])!=owned:raise RuntimeError('Other GPU probe PID '+f[0])
def run(name,command):
    guard();samples=[];start=time.monotonic()
    with (output/(name+'.log')).open('w') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                guard(process.pid);samples.append(dict(elapsed_seconds=time.monotonic()-start,guard_passed=True))
                if time.monotonic()-start>1200:raise TimeoutError('Owned sub-SIMD probe exceeded deadline')
                time.sleep(.5)
        except BaseException:
            if process.poll() is None:
                assert os.getpgid(process.pid)==process.pid;os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=15)
            raise
    (output/(name+'-monitor.json')).write_text(json.dumps(samples,indent=2)+'\n')
    statuses.append(dict(name=name,command=command,exit_code=process.returncode));(output/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n')
    assert process.returncode==0,name
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
native=r/'source/src/backends/metal/metal_context.mm'
paths=[native,s/'decode_wide_outputs_gpu.mm',s/'make_decode_matrix_probe.py',s/'make_decode_matrix_probe.py',s/'prepare_decode_tile_probe.py',Path(__file__),s/'compare_vllm_metal.py']
metadata=dict(scope='Real BF16 learned values and FP32 math; changed reduction order; component only',
    numerical_bounds='5e-6+5e-5*abs(control); both versus FP64 with gamma(ceil(K/32)+32)*sumabs+1e-7',
    source_sha256={str(p.relative_to(r/'source')):digest(p) for p in paths},
    hardware={k:subprocess.check_output(['/usr/sbin/sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize')})
(output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
for path in paths:
    copy=output/'source-snapshot'/path.relative_to(r/'source');copy.parent.mkdir(parents=True,exist_ok=True);copy.write_bytes(path.read_bytes())
(output/'before-processes.txt').write_text(snapshot());guard()
shader=output/'combined.metal';probe=output/'decode_matrix_gpu.mm';binary=output/'decode_matrix_gpu'
run('generate',[sys.executable,str(s/'make_decode_matrix_probe.py'),str(native),str(shader),str(probe)])

run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal',str(probe),'-o',str(binary)])
results=[];checks=[]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    assets=output/(model+'-assets');guard()
    run(model+'-prepare',[sys.executable,str(s/'prepare_decode_tile_probe.py'),str(r/'models'/model/'model.safetensors'),str(assets)])
    guard();time.sleep(5);guard();run(model+'-gpu',[str(binary),str(shader),str(assets)]);guard()
    manifest=json.loads((assets/'manifest.json').read_text());info={c['name']:c for c in manifest['cases']}
    rows=[json.loads(v) for v in (output/(model+'-gpu.log')).read_text().splitlines() if v.startswith('{')]
    quality=[v for v in rows if v['kind']=='check'];assert len(quality)==len(info)*4
    for q in quality:
        assert q['fp64_worst_ratio']<=1 and q['control_fp64_worst_ratio']<=1 and q['comparison_worst_ratio']<=1 and q['padding_preserved']
        checks.append(dict(model=model,**q))
        times=[v for v in rows if v['kind']=='timing' and all(v[k]==q[k] for k in ('matrix','rows','variant'))];assert len(times)==8
        control=statistics.median(v['gpu_ms_median'] for v in times if v['policy']=='control')
        candidate=statistics.median(v['gpu_ms_median'] for v in times if v['policy']=='candidate')
        results.append(dict(model=model,**info[q['matrix']],rows=q['rows'],variant=q['variant'],
                            control_gpu_ms=control,candidate_gpu_ms=candidate,component_speedup=control/candidate))
    (output/'summary.json').write_text(json.dumps(dict(scope=metadata['scope'],checks=checks,results=results),indent=2)+'\n')
(output/'after-processes.txt').write_text(snapshot());guard()
print(json.dumps(dict(checks=len(checks),timed_profiles=len(results),fp64_worst_ratio=max(c['fp64_worst_ratio'] for c in checks))),flush=True)
