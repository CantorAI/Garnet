"""Isolated real-weight FP32 decode work distribution: actual correctness and ABBA timing."""
import hashlib,json,os,signal,statistics,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve();output.mkdir(parents=True,exist_ok=False)
scripts=root/'source/test2026/metal';statuses=[]

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

import ast
background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')
nodes=[n for n in ast.parse((scripts/'compare_vllm_metal.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('verified_worldsense_service','snapshot','guard')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'owned_guard','exec'),globals())
strict_guard=guard
def guard(label):
    started=time.monotonic();waits=[]
    while True:
        try:
            strict_guard()
            for line in snapshot().splitlines():
                fields=line.split()
                if fields and fields[-1].lower().endswith(('_gpu','_probe')):
                    raise RuntimeError('Other GPU probe PID '+fields[0])
            break
        except RuntimeError as e:
            waits.append(dict(elapsed_seconds=time.monotonic()-started,reason=str(e)))
            (output/(label+'-wait.json')).write_text(json.dumps(waits,indent=2))
            if time.monotonic()-started>900:raise
            time.sleep(5)
    (output/(label+'-processes.txt')).write_text(snapshot())

def run(name,command):
    begin=time.monotonic()
    samples=[];interruption=None
    with (output/(name+'.log')).open('w') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                strict_guard()
                for line in snapshot().splitlines():
                    fields=line.split()
                    if fields and fields[-1].lower().endswith(('_gpu','_probe')):
                        if int(fields[0])!=process.pid:raise RuntimeError('Another GPU probe PID '+fields[0])
                samples.append(dict(elapsed_seconds=time.monotonic()-begin,guard_passed=True))
                if time.monotonic()-begin>1200:raise TimeoutError('Owned component job exceeded deadline')
                time.sleep(.5)
            code=process.returncode
        except Exception as error:
            interruption=str(error);code=1
            if process.poll() is None:
                assert os.getpgid(process.pid)==process.pid;os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    (output/(name+'-monitor.json')).write_text(json.dumps(samples,indent=2))
    statuses.append(dict(name=name,command=command,exit_code=code,interruption=interruption,diagnostic_wall_seconds=time.monotonic()-begin))
    (output/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n')
    if code:raise RuntimeError(name+' failed; retained log')

native=root/'source/src/backends/metal/metal_context.mm';shader=output/'combined.metal'
metadata=dict(scope='Native decode native-BF16 conversion/dot component experiment only, not serving or vLLM timing',
    precision='Exact BF16 weights, FP32 input/accumulation/output, serial unchanged/dot reassociated sums; independent FP64 reference; no quantization',
    hardware={key:subprocess.check_output(['/usr/sbin/sysctl','-n',key],text=True).strip() for key in ('machdep.cpu.brand_string','hw.memsize')},
    source_sha256={path.name:digest(path) for path in (native,scripts/'native_bfloat_decode_gpu.mm',scripts/'make_native_bfloat_decode_shader.py',scripts/'prepare_decode_tile_probe.py',scripts/'run_native_bfloat_decode_probe.py')})
(output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
run('shader',[sys.executable,str(scripts/'make_native_bfloat_decode_shader.py'),str(native),str(shader)])
binary=output/'native_bfloat_decode_gpu'
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal',str(scripts/'native_bfloat_decode_gpu.mm'),'-o',str(binary)])
summary=[];checks=[]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    guard(model+'-before');assets=output/(model+'-assets')
    run(model+'-prepare',[sys.executable,str(scripts/'prepare_decode_tile_probe.py'),str(root/'models'/model/'model.safetensors'),str(assets)])
    guard(model+'-gpu-before');run(model+'-gpu',[str(binary),str(shader),str(assets)]);guard(model+'-after')
    manifest=json.loads((assets/'manifest.json').read_text());info={case['name']:case for case in manifest['cases']}
    rows=[json.loads(line) for line in (output/(model+'-gpu.log')).read_text().splitlines() if line.startswith('{')]
    quality=[row for row in rows if row['kind']=='check'];assert len(quality)==len(info)*2*4
    checks.extend(dict(model=model,**row) for row in quality)
    for row in quality:
        times=[r for r in rows if r['kind']=='timing' and r['matrix']==row['matrix'] and r['rows']==row['rows'] and r['variant']==row['variant']];assert len(times)==8
        control=statistics.median(r['gpu_ms_median'] for r in times if r['policy']=='control')
        candidate=statistics.median(r['gpu_ms_median'] for r in times if r['policy']=='candidate')
        summary.append(dict(model=model,**info[row['matrix']],rows=row['rows'],variant=row['variant'],
            control_gpu_ms=control,candidate_gpu_ms=candidate,isolated_gpu_speedup=control/candidate,bits_exact=row['bits_exact'],max_abs_error=row['max_abs_error'],fp64_reference_passed=row['fp64_reference_passed'],worst_forward_error_ratio=row['worst_forward_error_ratio']))
    (output/'summary.json').write_text(json.dumps(dict(scope=metadata['scope'],results=summary,checks=checks),indent=2)+'\n')
print(json.dumps(dict(checks=len(checks),timed_profiles=len(summary),all_fp64_reference_passed=all(r['fp64_reference_passed'] for r in checks),worst_forward_error_ratio=max(r['worst_forward_error_ratio'] for r in checks))))
