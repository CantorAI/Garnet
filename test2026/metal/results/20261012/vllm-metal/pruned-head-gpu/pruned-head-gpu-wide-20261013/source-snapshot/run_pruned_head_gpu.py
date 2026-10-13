"""Exclusive native Metal head prototype with actual seed/bound/argmax costs.

Arguments: investigation root, new evidence directory. This is a head-only
component benchmark; no accelerated full-model generation executes.
"""
import hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve();output.mkdir(parents=True,exist_ok=False)
scripts=root/'source/test2026/metal';statuses=[]
wide='--wide' in sys.argv[3:]
reuse=Path(sys.argv[sys.argv.index('--reuse-assets')+1]).resolve() if '--reuse-assets' in sys.argv else None
if os.environ.get('GARNET_METAL_PROFILE_OPS')=='1':raise RuntimeError('Serialized diagnostics cannot count as speed')

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def guard(label):
    snapshot=subprocess.check_output(['/bin/ps','-axo','pid=,pcpu=,rss=,comm='],text=True)
    (output/(label+'-processes.txt')).write_text(snapshot)
    for line in snapshot.splitlines():
        fields=line.split();name=fields[-1].lower();pid=int(fields[0])
        if name.endswith('/xlang3'):
            current=pid;identified=False
            for _ in range(3):
                info=subprocess.run(['/bin/ps','-p',str(current),'-o','ppid=,args='],capture_output=True,text=True)
                parts=info.stdout.strip().split(None,1)
                if info.returncode or len(parts)!=2:break
                if '/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py' in parts[1]:identified=True;break
                current=int(parts[0])
                if current<=1:break
            libraries=subprocess.run(['/usr/sbin/lsof','-p',str(pid),'-Fn'],capture_output=True,text=True)
            if not identified or libraries.returncode or any(marker in libraries.stdout.lower() for marker in
                ('libgarnet','libmlx','libtorch','metal.framework','metalperformanceshaders.framework','libopenvino')):
                raise RuntimeError('Unverified xlang3 process PID '+str(pid))
        elif name.endswith(('/ollama','/vllm','/lossless_gpu_probe','/pruned_head_gpu')):raise RuntimeError('Other inference/GPU process PID '+str(pid))
    commands=subprocess.check_output(['/bin/ps','-axo','pid=,args='],text=True)
    for line in commands.splitlines():
        fields=line.strip().split(None,1)
        if len(fields)!=2 or int(fields[0])==os.getpid():continue
        if any(marker in fields[1] for marker in ('run_matched_vllm.py','vllm serve','vllm.entrypoints.openai','EngineCore::','frame_worker_vllm.py')):
            raise RuntimeError('Other inference worker PID '+fields[0])

def run(name,command):
    begin=time.monotonic()
    with (output/(name+'.log')).open('w') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=1200)
    statuses.append(dict(name=name,command=command,exit_code=result.returncode,diagnostic_wall_seconds=time.monotonic()-begin))
    (output/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n')
    if result.returncode:raise RuntimeError(name+' failed; retained log')

native=root/'source/src/backends/metal/metal_context.mm';text=native.read_text()
raw=text[text.index('struct BF16Params'):text.index('// BF16 weights are expanded')]
argmax=text[text.index('struct ArgmaxParams'):text.index('struct Params { ulong rows')]
shader=output/'combined.metal';shader.write_text('#include <metal_stdlib>\nusing namespace metal;\n'+raw+argmax+(scripts/'pruned_head_gpu.metal').read_text()+(scripts/'pruned_head_gpu_wide.metal').read_text())
metadata=dict(scope='Isolated native greedy-head prototype, not full-model inference or a vLLM speed claim',
    columns_per_simd=4 if wide else 1,reuse_assets=str(reuse) if reuse else None,
    hardware={key:subprocess.check_output(['/usr/sbin/sysctl','-n',key],text=True).strip() for key in ('machdep.cpu.brand_string','hw.memsize')},
    timed_pruned_operations=['8192-row exact native seed projection','GPU seed maximum and hidden residual norms','shared-row partial bounds and projection','unchanged native greedy argmax'],
    timed_control_operations=['complete unchanged native BF16 head projection','unchanged native greedy argmax'],
    excluded=['normalization common to both policies','weight norm preparation/upload','asset preparation','source compilation','CPU validation/readback'],
    raw_kernel_sha256=hashlib.sha256(raw.encode()).hexdigest(),argmax_kernel_sha256=hashlib.sha256(argmax.encode()).hexdigest(),
    source_sha256={path.name:digest(path) for path in (native,scripts/'pruned_head_gpu.mm',scripts/'pruned_head_gpu.metal',scripts/'pruned_head_gpu_wide.metal',scripts/'prepare_pruned_head_gpu.py',scripts/'run_pruned_head_gpu.py')})
(output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
binary=output/'pruned_head_gpu'
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal',str(scripts/'pruned_head_gpu.mm'),'-o',str(binary)])
summary=[]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    guard(model+'-before');assets=output/(model+'-assets')
    if reuse:
        origin=reuse/(model+'-assets');assets.mkdir();manifest=json.loads((origin/'manifest.json').read_text())
        assert digest(origin/'raw.bin')==manifest['raw_sha256'] and digest(origin/'remaining.bin')==manifest['remaining_norm_sha256']
        for query in manifest['queries']:
            assert digest(origin/(query['name']+'.input.bin'))==query['input_sha256']
            assert digest(origin/(query['name']+'.reference.bin'))==query['reference_sha256']
        (assets/'manifest.json').write_bytes((origin/'manifest.json').read_bytes())
        (assets/'reuse.json').write_text(json.dumps(dict(origin=str(origin),every_input_digest_verified=True),indent=2)+'\n');gpu_assets=origin
    else:
        run(model+'-prepare',[sys.executable,str(scripts/'prepare_pruned_head_gpu.py'),str(root),model,str(assets)]);gpu_assets=assets
    guard(model+'-gpu-before');run(model+'-gpu',[str(binary),str(shader),str(gpu_assets)]+(['wide'] if wide else []));guard(model+'-after')
    rows=[json.loads(line) for line in (output/(model+'-gpu.log')).read_text().splitlines() if line.startswith('{')]
    checks=[row for row in rows if row['kind']=='check'];assert len(checks)==17
    for check in checks:
        assert check['raw_native_bits_exact'] and check['retained_bits_exact'] and check['greedy_exact']
        times=[row for row in rows if row['kind']=='timing' and row['query']==check['query']];assert len(times)==8
        raw_time=statistics.median(row['gpu_ms_median'] for row in times if row['policy']=='raw')
        pruned_time=statistics.median(row['gpu_ms_median'] for row in times if row['policy']=='pruned')
        summary.append(dict(model=model,query=check['query'],batch=check['batch'],raw_gpu_ms=raw_time,pruned_gpu_ms=pruned_time,
            isolated_head_speedup=raw_time/pruned_time,physical_weight_fraction=check['physical_weight_fraction'],greedy_exact=True,retained_bits_exact=True))
    (output/'summary.json').write_text(json.dumps(dict(scope=metadata['scope'],results=summary),indent=2)+'\n')
print(json.dumps(dict(scope=metadata['scope'],results=summary),indent=2))
