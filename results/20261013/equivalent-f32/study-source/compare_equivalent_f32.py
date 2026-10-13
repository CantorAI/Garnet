"""Run Garnet and vLLM-Metal one process at a time, using exactly equivalent learned values at FP32 arithmetic.

Arguments: xlang3, native build, source, vLLM Python, evidence folder,
optional prefill policy and optional native control build for an ABBA check.
"""
import hashlib,json,os,platform,signal,subprocess,sys,time
from pathlib import Path

runtime,build,source,python,output=sys.argv[1:6];source=Path(source);root=source.parent;output=Path(output)
prefill_projection=sys.argv[6] if len(sys.argv)>6 else 'shared'
native_control=Path(sys.argv[7]) if len(sys.argv)>7 and sys.argv[7]!='-' else None
image_mode=len(sys.argv)>8 and sys.argv[8]=='vl'
if prefill_projection not in ('shared','mps'):raise ValueError('Unknown prefill projection policy')
if os.environ.get('GARNET_METAL_PROFILE_OPS')=='1':raise RuntimeError('Serialized GPU diagnostics cannot be used for speed comparisons')
output.mkdir(parents=True,exist_ok=True);scripts=source/'test2026/metal';statuses=[]
background_checks=[]
worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')

def verified_worldsense_service(pid):
    # A generic xlang3 HTTP service is not itself an inference runtime. Identify
    # this observed service by its actual ancestor script and loaded libraries;
    # unknown xlang3 jobs retain the conservative rejection.
    current=pid;identified=False
    for _ in range(3):
        result=subprocess.run(['/bin/ps','-p',str(current),'-o','ppid=,args='],capture_output=True,text=True)
        fields=result.stdout.strip().split(None,1)
        if result.returncode or len(fields)!=2:break
        if str(worldsense_script) in fields[1]:identified=True;break
        current=int(fields[0])
        if current<=1:break
    if not identified or not worldsense_script.is_file():return False
    source_text=worldsense_script.read_text(errors='replace').lower()
    if any(marker in source_text for marker in ('garnet','mlx','ollama','vllm','torch','qwen','metal')):return False
    libraries=subprocess.run(['/usr/sbin/lsof','-p',str(pid),'-Fn'],capture_output=True,text=True)
    if libraries.returncode:return False
    if any(marker in libraries.stdout.lower() for marker in
        ('libgarnet','libmlx','libtorch','metal.framework','metalperformanceshaders.framework','libopenvino')):return False
    background_checks.append(dict(pid=pid,role='WorldSense service without inference/GPU libraries',
        ancestor_script=str(worldsense_script),script_sha256=hashlib.sha256(worldsense_script.read_bytes()).hexdigest(),
        timestamp=time.time()))
    (output/'background_service_checks.json').write_text(json.dumps(background_checks,indent=2)+'\n')
    return True

def snapshot():
    # Process names only: never dump command lines, environment or credentials.
    return subprocess.check_output(['/bin/ps','-axo','pid=,pcpu=,rss=,comm='],text=True)

def guard():
    lines=snapshot().splitlines();active=[]
    for line in lines:
        name=line.split()[-1].lower() if line.split() else ''
        if name.endswith('/xlang3'):
            if not verified_worldsense_service(int(line.split()[0])):active.append(line)
        elif name.endswith('/ollama') or name.endswith('/vllm'):
            active.append(line)
    # Python workers need argument inspection to identify their role. Never
    # print or save their arguments, which may belong to unrelated user jobs.
    commands=subprocess.check_output(['/bin/ps','-axo','pid=,args='],text=True)
    for line in commands.splitlines():
        fields=line.strip().split(None,1)
        if len(fields)!=2 or int(fields[0])==os.getpid():continue
        command=fields[1]
        if 'background_comparison.py' in command or 'compare_vllm_metal.py' in command:continue
        if any(marker in command for marker in ('run_matched_vllm.py','vllm serve','vllm.entrypoints.openai','EngineCore::',
            'setup_vllm_comparison.py','frame_worker_vllm.py')):
            active.append('Inference worker PID '+fields[0])
    if active:raise RuntimeError('Another inference runtime is active: '+json.dumps(active))

def save_status():
    (output/'job_statuses.json').write_text(json.dumps(statuses,indent=2)+'\n')

manifest=dict(platform=platform.platform(),native_build=build,native_library_sha256=hashlib.file_digest((Path(build)/'libgarnet.dylib').open('rb'),'sha256').hexdigest(),
    policy='Only one runtime process alive. Identical BF16 checkpoint and native prompt token IDs. Prefix cache/speculation disabled.',models={})
manifest['hardware']={key:subprocess.check_output(['/usr/sbin/sysctl','-n',key],text=True).strip()
    for key in ('machdep.cpu.brand_string','hw.memsize','hw.physicalcpu','hw.logicalcpu')}
manifest['runner_sha256']={name:hashlib.sha256((scripts/name).read_bytes()).hexdigest()
    for name in ('run_matched_garnet.py','run_matched_vllm.py','benchmark_matched_process.py','summarize_vllm_comparison.py')}
manifest['runner_sha256']['compare_equivalent_f32.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
if image_mode:
    manifest['comparison_kind']='VL common CPU frontend and persistent workers'
    manifest['runner_sha256'].update({name:hashlib.sha256((scripts/name).read_bytes()).hexdigest() for name in
        ('frame_preprocessing.py','frame_worker_garnet.py','frame_worker_vllm.py','run_matched_frame_process.py')})
if native_control is not None:
    manifest['comparison_kind']='native ABBA; no vLLM speed claim'
    manifest['native_control_build']=str(native_control)
    manifest['native_control_library_sha256']=hashlib.file_digest((native_control/'libgarnet.dylib').open('rb'),'sha256').hexdigest()
cleanup=Path(python).parent.parent/'lib/python3.12/site-packages/vllm/distributed/parallel_state.py'
manifest['comparator_cleanup_source_sha256']=hashlib.sha256(cleanup.read_bytes()).hexdigest()
model_prompts=[('Qwen3.5-0.8B','Describe the main objects in this image in one sentence.')] if image_mode else [
    ('Qwen3-0.6B','What is the capital of France? Answer briefly.'),
    ('Qwen2.5-Coder-0.5B-Instruct','Write a Python function add(a, b) that returns their sum. Output only code.')]
for name,_ in model_prompts:
    model=root/'models'/name
    manifest['models'][name]={file:dict(bytes=(model/file).stat().st_size,sha256=hashlib.file_digest((model/file).open('rb'),'sha256').hexdigest())
        for file in (('model.safetensors','config.json','tokenizer.json','tokenizer_config.json','preprocessor_config.json') if image_mode else ('model.safetensors','config.json','tokenizer.json','tokenizer_config.json'))}
if image_mode:
    manifest['image_sha256']=hashlib.file_digest((root/'images/RealWorld-04.png').open('rb'),'sha256').hexdigest()
assert native_control is None,'This precision study compares two runtimes, not native builds'
manifest['policy']='Sequential extra FP32 precision study, not a replacement for the default BF16 comparison. Native original BF16 storage / FP32 operations versus lossless F32 checkpoint values / FP32 comparator.'
manifest['counts_as_default_bf16_speedup']=False
manifest['equivalent_comparator_models']={}
for name,_ in model_prompts:
    folder=root/'models-exact-f32'/name;proof_path=folder/'exact-value-equivalence.json';proof=json.loads(proof_path.read_text())
    assert proof['all_values_bit_exact'] and all(v['all_values_bit_exact'] for v in proof['tensors'])
    assert proof['source_sha256']==manifest['models'][name]['model.safetensors']['sha256']
    assert hashlib.file_digest((folder/'model.safetensors').open('rb'),'sha256').hexdigest()==proof['destination_sha256']
    manifest['equivalent_comparator_models'][name]=dict(path=str(folder),checkpoint_sha256=proof['destination_sha256'],
        equivalence_proof_sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest(),all_learned_values_bit_exact=True,
        assets={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.json')})
    (output/(name+'-exact-value-equivalence.json')).write_bytes(proof_path.read_bytes())
(output/'metadata.json').write_text(json.dumps(manifest,indent=2)+'\n')

for batch in (1,2):
    for name,prompt in model_prompts:
        case=dict(prompt=prompt,batch=batch,max_tokens=32 if image_mode else 64,capacity=256,chunk=16,requests=3,prefill_projection=prefill_projection)
        case['comparator_dtype']='float32'
        case['comparator_gpu_memory_utilization']=.45 if image_mode else .30
        if image_mode:case['image_metadata']=json.loads((root/'images'/('RealWorld-04.batch'+str(batch)+'.json')).read_text())
        model=root/'models'/name;casefile=output/(name+'-b'+str(batch)+'-case.json')
        casefile.write_text(json.dumps(case,indent=2)+'\n')
        native_log=None
        for repeat in range(2):
            order=('garnet','vllm-metal') if native_control is None else (('garnet','garnet-control') if repeat==0 else ('garnet-control','garnet'))
            for backend in order:
                guard();folder=output/(name+'-b'+str(batch)+'-'+backend+'-'+str(repeat))
                folder.mkdir(parents=True,exist_ok=True)
                (folder/'preflight.txt').write_text(snapshot())
                comparator_model=root/'models-exact-f32'/name
                if image_mode:
                    native_backend=backend in ('garnet','garnet-control')
                    executable=runtime if native_backend else python
                    selected_build=str(native_control) if backend=='garnet-control' else build
                    worker_backend='garnet' if native_backend else backend
                    command=[python,str(scripts/'run_matched_frame_process.py'),executable,selected_build,str(source),str(model if native_backend else comparator_model),str(casefile),worker_backend,str(folder/'protocol')]
                elif backend in ('garnet','garnet-control'):
                    selected_build=build if backend=='garnet' else str(native_control)
                    command=[runtime,str(scripts/'run_matched_garnet.py'),selected_build,str(source),str(model),str(casefile)]
                else:command=[python,str(scripts/'run_matched_vllm.py'),str(comparator_model),str(casefile),str(native_log)]
                command=[sys.executable,str(scripts/'benchmark_matched_process.py'),*command]
                print('Starting exclusive comparison',folder.name,flush=True);begin=time.monotonic()
                timed_out=False
                with (folder/'output.log').open('w') as stream:
                    process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    try:process.wait(timeout=900)
                    except subprocess.TimeoutExpired:
                        # Kill only the session created above, including workers;
                        # a timed-out wrapper must not leave inference running.
                        timed_out=True
                        if process.poll() is None:
                            assert os.getpgid(process.pid)==process.pid
                            os.killpg(process.pid,signal.SIGTERM)
                            try:process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid,signal.SIGKILL);process.wait()
                status=dict(name=folder.name,command=command,exit_code=process.returncode,
                    timed_out=timed_out,process_seconds=time.monotonic()-begin)
                statuses.append(status);(folder/'exit.json').write_text(json.dumps(status,indent=2)+'\n');save_status()
                guard();(folder/'postflight.txt').write_text(snapshot())
                print('Completed',folder.name,'exit',process.returncode,flush=True)
                if timed_out or process.returncode:raise SystemExit(process.returncode or 1)
                if backend=='garnet':
                    native_log=folder/'output.log'
                    if image_mode:
                        case['native_prompt']=json.loads((folder/'protocol/ready.json').read_text())
                        casefile.write_text(json.dumps(case,indent=2)+'\n')
