"""Joined exact-half MPS gate/up with producer and consumer; component only."""
import ast,hashlib,json,os,signal,statistics,subprocess,sys,time
from pathlib import Path
r=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve();output.mkdir(parents=True,exist_ok=False)
s=r/'source/test2026/metal';statuses=[];background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((s/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals())
def digest(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def check(owned=None):
    guard()
    for line in snapshot().splitlines():
        f=line.split()
        if f and f[-1].lower().endswith(('_gpu','_probe')) and int(f[0])!=owned:raise RuntimeError('Other GPU probe PID '+f[0])
def run(name,command):
    check();begin=time.monotonic();samples=[];interruption=None
    with (output/(name+'.log')).open('w') as log:
        p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while p.poll() is None:
                check(p.pid);samples.append(dict(elapsed_seconds=time.monotonic()-begin,guard_passed=True))
                if time.monotonic()-begin>300:raise TimeoutError('Owned component deadline')
                time.sleep(.5)
            code=p.returncode
        except Exception as e:
            interruption=str(e);code=1
            if p.poll() is None:
                assert os.getpgid(p.pid)==p.pid;os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    statuses.append(dict(name=name,command=command,exit_code=code,interruption=interruption,wall_seconds=time.monotonic()-begin))
    (output/'statuses.json').write_text(json.dumps(statuses,indent=2));(output/(name+'-monitor.json')).write_text(json.dumps(samples,indent=2))
    check();assert code==0,(name,code,interruption)
source=s/'joined_mps_pair_gpu.mm';binary=output/'joined_mps_pair_gpu'
metadata=dict(scope=__doc__,source_sha256={str(p.relative_to(r/'source')):digest(p) for p in (source,Path(__file__),s/'prepare_prefill_tile_probe.py',s/'compare_vllm_metal.py',r/'source/src/backends/metal/exact_half_codec.h')},hardware={k:subprocess.check_output(['/usr/sbin/sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize')})
(output/'metadata.json').write_text(json.dumps(metadata,indent=2))
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders',str(source),'-o',str(binary)])
results=[];checks=[]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    assets=output/(model+'-assets');run(model+'-prepare',[sys.executable,str(s/'prepare_prefill_tile_probe.py'),str(r/'models'/model/'model.safetensors'),str(assets)])
    run(model+'-gpu',[str(binary),str(assets)])
    records=[json.loads(line) for line in (output/(model+'-gpu.log')).read_text().splitlines() if line.startswith('{')]
    quality=[v for v in records if v['kind']=='check'];assert len(quality)==8 and all(v['padding_preserved'] and v['worst_ratio']<=1 and v['consumer_worst_ratio']<=1 and v['generation_checks']==16 for v in quality)
    checks.extend(dict(model=model,**v) for v in quality)
    for v in quality:
        times=[t for t in records if t['kind']=='timing' and t['rows']==v['rows']];assert len(times)==8
        control=statistics.median(t['gpu_completion_span_ms'] for t in times if not t['joined']);candidate=statistics.median(t['gpu_completion_span_ms'] for t in times if t['joined'])
        control_wall=statistics.median(t['host_wall_ms'] for t in times if not t['joined']);candidate_wall=statistics.median(t['host_wall_ms'] for t in times if t['joined'])
        results.append(dict(model=model,rows=v['rows'],control_gpu_completion_span_ms=control,candidate_gpu_completion_span_ms=candidate,component_completion_speedup=control/candidate,control_host_wall_ms=control_wall,candidate_host_wall_ms=candidate_wall,host_speedup=control_wall/candidate_wall,observed_interval_overlap_ms=statistics.median(t['interval_overlap_ms'] for t in times if t['joined']),bits_exact=v['bits_exact']))
    (output/'summary.json').write_text(json.dumps(dict(scope=__doc__,checks=checks,results=results,producer_consumer_event_cost_included=True,serving_changed=False),indent=2))
    print('Completed independent MPS pairs',model,'checks',len(quality),flush=True)
print(json.dumps(dict(checks=len(checks),bits_exact=all(v['bits_exact'] for v in checks))),flush=True)
