"""Sequential actual native captures and offline partial-bound feasibility.

Arguments: investigation root, new evidence directory. All timing is
diagnostic setup/analysis and is excluded from inference-speed comparisons.
"""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve();output.mkdir(parents=True,exist_ok=False)
scripts=root/'source/test2026/metal';build=root/'build-state-handoff'
runtime=Path('/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3');statuses=[]

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
        elif name.endswith(('/ollama','/vllm','/lossless_gpu_probe')):raise RuntimeError('Other inference/GPU process PID '+str(pid))
    commands=subprocess.check_output(['/bin/ps','-axo','pid=,args='],text=True)
    for line in commands.splitlines():
        fields=line.strip().split(None,1)
        if len(fields)!=2 or int(fields[0])==os.getpid():continue
        if any(marker in fields[1] for marker in ('run_matched_vllm.py','vllm serve','vllm.entrypoints.openai','EngineCore::','frame_worker_vllm.py')):
            raise RuntimeError('Other inference worker PID '+fields[0])

def run(name,command):
    begin=time.monotonic()
    environment=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    with (output/(name+'.log')).open('w') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
        try:code=process.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            assert os.getpgid(process.pid)==process.pid
            os.killpg(process.pid,15)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,9);process.wait()
            code=process.returncode or 1
    statuses.append(dict(name=name,command=command,exit_code=code,diagnostic_wall_seconds=time.monotonic()-begin))
    (output/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n')
    if code:raise RuntimeError(name+' failed; retained diagnostic log')

metadata=dict(scope='Actual full native inference captures plus offline counterfactual partial bounds; no pruning speed claim',
    build=str(build),native_library_sha256=digest(build/'libgarnet.dylib'),
    source_sha256={name:digest(scripts/name) for name in ('capture_head_pruning_queries.py','analyze_partial_head_bounds.py','run_partial_head_probe.py')},
    hardware={key:subprocess.check_output(['/usr/sbin/sysctl','-n',key],text=True).strip() for key in ('machdep.cpu.brand_string','hw.memsize')},models={})
(output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');summary=[]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    checkpoint=root/'models'/model/'model.safetensors';metadata['models'][model]=dict(bytes=checkpoint.stat().st_size,sha256=digest(checkpoint))
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for batch in (1,2):
        name=model+'-b'+str(batch);folder=output/name;folder.mkdir()
        vl=model=='Qwen3.5-0.8B'
        baseline=root/'logs'/('vllm-matched-vl-file-events-20261012' if vl else 'vllm-matched-state-handoff-20261012')/'jobs'
        case=json.loads((baseline/(name+'-case.json')).read_text());(folder/'case.json').write_text(json.dumps(case,indent=2)+'\n')
        if vl:
            assert digest(Path(case['image_metadata']['patch_file']))==case['image_metadata']['patch_sha256']
            expected=json.loads((baseline/(name+'-garnet-0')/'protocol/command.json.event-0-frame_result.json').read_text())
        else:
            lines=(baseline/(name+'-garnet-0')/'output.log').read_text().splitlines()
            expected=next(json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('matched_result '))
        guard(name+'-before')
        run(name+'-capture',[str(runtime),str(scripts/'capture_head_pruning_queries.py'),str(build),str(root/'source'),str(checkpoint.parent),str(folder/'case.json'),str(folder)])
        guard(name+'-after')
        capture=json.loads((folder/'queries.json').read_text())
        assert capture['token_ids']==expected['token_ids'] and capture['prompt_token_ids']==expected['prompt_token_ids']
        assert capture['finished']==expected['finished']
        (folder/'output_validation.json').write_text(json.dumps(dict(exact_prompt=True,exact_complete_native_output=True,
            baseline=str(baseline),captured_queries=len(capture['queries']),batch=batch),indent=2)+'\n')
        run(name+'-analyze',[sys.executable,str(scripts/'analyze_partial_head_bounds.py'),str(checkpoint),str(folder/'queries.json'),str(folder/'report.json')])
        report=json.loads((folder/'report.json').read_text())
        summary.append({key:report[key] for key in ('model','batch','queries','head_shape','all_partial_bounds_cover_every_actual_gpu_logit','actual_winners_preserved','aggregates')})
        (output/'summary.json').write_text(json.dumps(dict(scope=metadata['scope'],results=summary),indent=2)+'\n')
        print('Completed native capture and offline bounds',name,flush=True)
