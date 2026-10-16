"""Isolated MPS prefill layout; no serving or vLLM throughput claim."""
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
                if fields and fields[-1].lower().endswith(('/decode_wide_outputs_gpu','/fused_mlp_decode_gpu','/lossless_gpu_probe','/decode_occupancy_gpu','/transpose_half_prefill_probe','/decode_accumulators_gpu','/decode_unroll_gpu','/pruned_head_gpu','/prefill_tile_gpu')):
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
                    if fields and fields[-1].lower().endswith(('/transpose_half_prefill_probe','/decode_accumulators_gpu','/decode_unroll_gpu','/decode_single_output_gpu','/decode_wide_outputs_gpu','/fused_mlp_decode_gpu','/lossless_gpu_probe','/decode_occupancy_gpu','/pruned_head_gpu','/prefill_tile_gpu','/split_half_prefill_probe','/lossless_block_gpu_probe','/parallel_delta_gpu')):
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

metadata=dict(scope='Exact scaled-half right-matrix transposition, FP32 activations/output; component only',source_sha256={name:digest(root/'source'/name) for name in ('test2026/metal/run_transpose_half_prefill_probe.py','test2026/metal/transpose_half_prefill_probe.mm','test2026/metal/prepare_prefill_tile_probe.py','src/backends/metal/exact_half_codec.h')})
(output/'metadata.json').write_text(json.dumps(metadata,indent=2))
binary=output/'transpose_half_prefill_probe'
guard('compile-before')
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal','-framework','MetalPerformanceShaders',str(scripts/'transpose_half_prefill_probe.mm'),'-o',str(binary)])
results=[];checks=[];excluded=[]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    guard(model+'-before');assets=output/(model+'-assets')
    run(model+'-prepare',[sys.executable,str(scripts/'prepare_prefill_tile_probe.py'),str(root/'models'/model/'model.safetensors'),str(assets)])
    guard(model+'-gpu-before');run(model+'-gpu',[str(binary),str(assets)]);guard(model+'-after')
    manifest=json.loads((assets/'manifest.json').read_text());info={c['name']:c for c in manifest['cases']}
    records=[json.loads(v) for v in (output/(model+'-gpu.log')).read_text().splitlines() if v.startswith('{')]
    omitted=[v for v in records if v['kind']=='excluded'];quality=[v for v in records if v['kind']=='check']
    assert len(quality)==(len(info)-len(omitted))*8 and quality
    assert all(v['worst_ratio']<=1 and v['encoded_weight_bits_equal'] and v['padding_preserved'] for v in quality)
    excluded.extend(dict(model=model,**info[v['name']],reason=v['reason']) for v in omitted)
    checks.extend(dict(model=model,**v) for v in quality)
    for value in quality:
        times=[v for v in records if v['kind']=='timing' and v['name']==value['name'] and v['rows']==value['rows']];assert len(times)==8
        control=statistics.median(v['gpu_ms'] for v in times if not v['candidate']);candidate=statistics.median(v['gpu_ms'] for v in times if v['candidate'])
        results.append(dict(model=model,**info[value['name']],rows=value['rows'],control_gpu_ms=control,candidate_gpu_ms=candidate,isolated_gpu_speedup=control/candidate,worst_ratio=value['worst_ratio']))
    (output/'summary.json').write_text(json.dumps(dict(scope=metadata['scope'],results=results,checks=checks,excluded=excluded),indent=2))
    print('Completed component',model,'checks',len(quality),'excluded',len(omitted),flush=True)
print(json.dumps(dict(checks=len(checks),excluded=len(excluded),worst_ratio=max(v['worst_ratio'] for v in checks))))
