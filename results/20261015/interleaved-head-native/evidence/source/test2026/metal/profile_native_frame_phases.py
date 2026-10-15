"""Sequential actual loaded-parameter dtype audit, not a speed comparison."""
import hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve();output.mkdir(parents=True,exist_ok=False)
scripts=root/'source/test2026/metal';statuses=[]

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
        elif name.endswith(('/ollama','/vllm','/lossless_gpu_probe','/pruned_head_gpu','/decode_occupancy_gpu')):raise RuntimeError('Other inference/GPU process PID '+str(pid))
    commands=subprocess.check_output(['/bin/ps','-axo','pid=,args='],text=True)
    for line in commands.splitlines():
        fields=line.strip().split(None,1)
        if len(fields)!=2 or int(fields[0])==os.getpid():continue
        if any(marker in fields[1] for marker in ('run_matched_vllm.py','vllm serve','vllm.entrypoints.openai','EngineCore::','frame_worker_vllm.py')):
            raise RuntimeError('Other inference worker PID '+fields[0])

def run(name,command):
    begin=time.monotonic()
    with (output/(name+'.log')).open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=1200)
    statuses.append(dict(name=name,command=command,exit_code=result.returncode,diagnostic_wall_seconds=time.monotonic()-begin))
    (output/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n')
    if result.returncode:raise RuntimeError(name+' failed; retained log')

frontend=str(root/'comparators/vllm/venv/bin/python');runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3'
build=root/'build-mps-vision-scores'
metadata=dict(scope='Native serving phase counters; no serialized GPU operations',library_sha256=digest(build/'libgarnet.dylib'),worker_sha256=digest(scripts/'frame_worker_garnet.py'))
(output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
records=[];groups=[]
for batch in (1,2):
    label='Qwen3.5-0.8B-b'+str(batch)
    baseline=root/'logs/native-frame-ab-mps-vision-confirm-20261013/jobs'
    case=json.loads((baseline/(label+'-case.json')).read_text());casefile=output/(label+'-case.json');casefile.write_text(json.dumps(case,indent=2)+'\n')
    expected=json.loads((baseline/(label+'-garnet-0')/'protocol/command.json.event-0-frame_result.json').read_text())
    for repeat in range(2):
        name=label+'-'+str(repeat);folder=output/name;folder.mkdir();guard(name+'-before')
        run(name,[frontend,str(scripts/'run_matched_frame_process.py'),runtime,str(build),str(root/'source'),str(root/'models/Qwen3.5-0.8B'),str(casefile),'garnet',str(folder/'protocol')])
        guard(name+'-after')
        values=[json.loads(line.split(' ',1)[1]) for line in (output/(name+'.log')).read_text().splitlines() if line.startswith('matched_frame_result ')];assert len(values)==3
        for value in values:
            assert value['token_ids']==expected['token_ids'] and value['finished']==expected['finished']
            assert value['prompt_token_ids']==expected['prompt_token_ids']
            value['repeat']=repeat;records.append(value)
    warm=[v for v in records if v['case']['batch']==batch and v['request']>0]
    group=dict(batch=batch,exact_complete_outputs=True,requests=len(warm),
        stages={key:statistics.mean(v[key] for v in warm) for key in ('ttft_seconds','vision_seconds','assembly_seconds','prefill_wall_seconds','prefill_inference_ms')},
        prefill_profile={key:statistics.mean(v['prefill_profile'][key] for v in warm) for key in warm[0]['prefill_profile']})
    groups.append(group);(output/'summary.json').write_text(json.dumps(dict(scope=metadata['scope'],groups=groups,records=records),indent=2)+'\n')
    print(json.dumps(group),flush=True)
