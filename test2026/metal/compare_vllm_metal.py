"""Run Garnet and vLLM-Metal one process at a time, using identical checkpoint paths.

Arguments: xlang3, native build, source, vLLM Python, evidence folder.
"""
import hashlib,json,os,platform,signal,subprocess,sys,time
from pathlib import Path

runtime,build,source,python,output=sys.argv[1:];source=Path(source);root=source.parent;output=Path(output)
output.mkdir(parents=True,exist_ok=True);scripts=source/'test2026/metal';statuses=[]

def snapshot():
    # Process names only: never dump command lines, environment or credentials.
    return subprocess.check_output(['/bin/ps','-axo','pid=,pcpu=,rss=,comm='],text=True)

def guard():
    lines=snapshot().splitlines();active=[]
    for line in lines:
        name=line.split()[-1].lower() if line.split() else ''
        if name.endswith('/xlang3') or name.endswith('/ollama') or name.endswith('/vllm'):
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
            'setup_vllm_comparison.py')):
            active.append('Inference worker PID '+fields[0])
    if active:raise RuntimeError('Another inference runtime is active: '+json.dumps(active))

def save_status():
    (output/'job_statuses.json').write_text(json.dumps(statuses,indent=2)+'\n')

manifest=dict(platform=platform.platform(),native_build=build,native_library_sha256=hashlib.file_digest((Path(build)/'libgarnet.dylib').open('rb'),'sha256').hexdigest(),
    policy='Only one runtime process alive. Identical BF16 checkpoint and native prompt token IDs. Prefix cache/speculation disabled.',models={})
manifest['hardware']={key:subprocess.check_output(['/usr/sbin/sysctl','-n',key],text=True).strip()
    for key in ('machdep.cpu.brand_string','hw.memsize','hw.physicalcpu','hw.logicalcpu')}
for name in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct'):
    model=root/'models'/name
    manifest['models'][name]={file:dict(bytes=(model/file).stat().st_size,sha256=hashlib.file_digest((model/file).open('rb'),'sha256').hexdigest())
        for file in ('model.safetensors','config.json','tokenizer.json','tokenizer_config.json')}
(output/'metadata.json').write_text(json.dumps(manifest,indent=2)+'\n')

for batch in (1,2):
    for name,prompt in [('Qwen3-0.6B','What is the capital of France? Answer briefly.'),
        ('Qwen2.5-Coder-0.5B-Instruct','Write a Python function add(a, b) that returns their sum. Output only code.')]:
        case=dict(prompt=prompt,batch=batch,max_tokens=64,capacity=256,chunk=16,requests=3)
        model=root/'models'/name;casefile=output/(name+'-b'+str(batch)+'-case.json')
        casefile.write_text(json.dumps(case,indent=2)+'\n')
        native_log=None
        for repeat in range(2):
            for backend in ('garnet','vllm-metal'):
                guard();folder=output/(name+'-b'+str(batch)+'-'+backend+'-'+str(repeat))
                folder.mkdir(parents=True,exist_ok=True)
                (folder/'preflight.txt').write_text(snapshot())
                if backend=='garnet':command=[runtime,str(scripts/'run_matched_garnet.py'),build,str(source),str(model),str(casefile)]
                else:command=[python,str(scripts/'run_matched_vllm.py'),str(model),str(casefile),str(native_log)]
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
                if backend=='garnet':native_log=folder/'output.log'
