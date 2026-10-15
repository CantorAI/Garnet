"""Isolated real-weight FP32 decode work distribution: actual correctness and ABBA timing."""
import hashlib,json,os,statistics,subprocess,sys,time
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
                if fields and fields[-1].lower().endswith(('/decode_wide_outputs_gpu','/fused_mlp_decode_gpu','/lossless_gpu_probe','/decode_occupancy_gpu','/decode_unroll_gpu','/pruned_head_gpu','/prefill_tile_gpu')):
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
    with (output/(name+'.log')).open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=1200)
    statuses.append(dict(name=name,command=command,exit_code=result.returncode,diagnostic_wall_seconds=time.monotonic()-begin))
    (output/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n')
    if result.returncode:raise RuntimeError(name+' failed; retained log')

native=root/'source/src/backends/metal/metal_context.mm';shader=output/'combined.metal'
metadata=dict(scope='Native decode dot-loop unroll component experiment only, not serving or vLLM timing',
    precision='Unchanged exact BF16 weights, FP32 input/accumulation/output; no quantization',
    hardware={key:subprocess.check_output(['/usr/sbin/sysctl','-n',key],text=True).strip() for key in ('machdep.cpu.brand_string','hw.memsize')},
    source_sha256={path.name:digest(path) for path in (native,scripts/'decode_unroll_gpu.mm',scripts/'make_decode_unroll_shader.py',scripts/'prepare_decode_tile_probe.py',scripts/'run_decode_unroll_probe.py')})
(output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
run('shader',[sys.executable,str(scripts/'make_decode_unroll_shader.py'),str(native),str(shader)])
binary=output/'decode_unroll_gpu'
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal',str(scripts/'decode_unroll_gpu.mm'),'-o',str(binary)])
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
            control_gpu_ms=control,candidate_gpu_ms=candidate,isolated_gpu_speedup=control/candidate,bits_exact=row['bits_exact'],max_abs_error=row['max_abs_error']))
    (output/'summary.json').write_text(json.dumps(dict(scope=metadata['scope'],results=summary,checks=checks),indent=2)+'\n')
print(json.dumps(dict(checks=len(checks),timed_profiles=len(summary),all_bits_exact=all(r['bits_exact'] for r in checks))))
