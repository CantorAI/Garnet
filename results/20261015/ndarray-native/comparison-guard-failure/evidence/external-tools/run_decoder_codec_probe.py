"""Measure exact decoder packing one model process at a time."""
import hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve();output.mkdir(parents=True,exist_ok=False)
scripts=root/'source/test2026/metal';statuses=[]

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def guard(label):
    snapshot=subprocess.check_output(['/bin/ps','-axo','pid=,pcpu=,rss=,comm='],text=True);(output/(label+'-processes.txt')).write_text(snapshot)
    for line in snapshot.splitlines():
        fields=line.split();name=fields[-1].lower();pid=int(fields[0])
        if name.endswith('/xlang3'):
            current=pid;identified=False
            for _ in range(3):
                info=subprocess.run(['/bin/ps','-p',str(current),'-o','ppid=,args='],capture_output=True,text=True);parts=info.stdout.strip().split(None,1)
                if info.returncode or len(parts)!=2:break
                if '/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py' in parts[1]:identified=True;break
                current=int(parts[0])
                if current<=1:break
            libraries=subprocess.run(['/usr/sbin/lsof','-p',str(pid),'-Fn'],capture_output=True,text=True)
            if not identified or libraries.returncode or any(marker in libraries.stdout.lower() for marker in ('libgarnet','libmlx','libtorch','metal.framework','metalperformanceshaders.framework','libopenvino')):raise RuntimeError('Unverified xlang3 PID '+str(pid))
        elif name.endswith(('/ollama','/vllm','/lossless_gpu_probe','/pruned_head_gpu','/prefill_tile_gpu')):raise RuntimeError('Other inference/GPU PID '+str(pid))
    commands=subprocess.check_output(['/bin/ps','-axo','pid=,args='],text=True)
    for line in commands.splitlines():
        fields=line.strip().split(None,1)
        if len(fields)!=2 or int(fields[0])==os.getpid():continue
        if any(marker in fields[1] for marker in ('run_matched_vllm.py','vllm serve','vllm.entrypoints.openai','EngineCore::','frame_worker_vllm.py')):raise RuntimeError('Other inference worker PID '+fields[0])

def run(name,command):
    begin=time.monotonic()
    with (output/(name+'.log')).open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=1200)
    statuses.append(dict(name=name,command=command,exit_code=result.returncode,diagnostic_wall_seconds=time.monotonic()-begin))
    (output/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n')
    if result.returncode:raise RuntimeError(name+' failed; retained log')

native=root/'source/src/backends/metal/metal_context.mm';text=native.read_text();raw=text[text.index('struct BF16Params'):text.index('// BF16 weights are expanded')]
shader=output/'combined.metal';shader.write_text('#include <metal_stdlib>\nusing namespace metal;\n'+raw+(scripts/'lossless_gpu_probe.metal').read_text())
metadata=dict(scope='Exact decoder matrix-vector component test; no full-model speed claim',
    hardware={key:subprocess.check_output(['/usr/sbin/sysctl','-n',key],text=True).strip() for key in ('machdep.cpu.brand_string','hw.memsize')},
    source_sha256={path.name:digest(path) for path in (native,scripts/'lossless_gpu_probe.mm',scripts/'lossless_gpu_probe.metal',scripts/'prepare_decoder_codec_probe.py',scripts/'prepare_lossless_gpu_probe.py',scripts/'probe_lossless_nibble_bf16_codec.py',scripts/'run_decoder_codec_probe.py')},
    raw_kernel_sha256=hashlib.sha256(raw.encode()).hexdigest(),codec='lossless nibble; one output column per SIMD group')
(output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');binary=output/'lossless_gpu_probe'
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal',str(scripts/'lossless_gpu_probe.mm'),'-o',str(binary)])
summary=[]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    guard(model+'-before');assets=output/(model+'-assets')
    run(model+'-prepare',[sys.executable,str(scripts/'prepare_decoder_codec_probe.py'),str(root/'models'/model/'model.safetensors'),str(assets)])
    guard(model+'-gpu-before');run(model+'-gpu',[str(binary),str(shader),str(assets),'single']);guard(model+'-after')
    manifest=json.loads((assets/'manifest.json').read_text());rows=[json.loads(line) for line in (output/(model+'-gpu.log')).read_text().splitlines() if line.startswith('{')]
    assert len([r for r in rows if r['kind']=='roundtrip'])==len(manifest['cases'])
    for case in manifest['cases']:
        for batch in (1,2):
            check=next(r for r in rows if r['kind']=='projection_check' and r['case']==case['name'] and r['batch']==batch);assert check['output_bits_exact']
            times=[r for r in rows if r['kind']=='timing' and r['case']==case['name'] and r['batch']==batch];assert len(times)==16
            control=statistics.median(r['gpu_ms_median'] for r in times if r['policy']=='raw');candidate=statistics.median(r['gpu_ms_median'] for r in times if r['policy']=='packed')
            summary.append(dict(model=model,**case,batch=batch,control_gpu_ms=control,packed_gpu_ms=candidate,isolated_gpu_speedup=control/candidate,output_bits_exact=True))
    (output/'summary.json').write_text(json.dumps(dict(scope=metadata['scope'],results=summary),indent=2)+'\n')
print(json.dumps(dict(matrices=len(summary)//2,checks=len(summary),all_bits_exact=True)))
