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

frontend=str(root/'comparators/vllm/venv/bin/python')
metadata=dict(scope='Read-only actual loaded MLX model dtype audit, not speed evidence',
    runner_sha256=digest(Path(__file__)),worker_sha256=digest(scripts/'audit_comparator_dtypes.py'))
(output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
reports=[]
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    for dtype in ('bfloat16','float32'):
        label=model+'-'+dtype;guard(label+'-before')
        run(label,[frontend,str(scripts/'audit_comparator_dtypes.py'),str(root/'models'/model),dtype])
        guard(label+'-after')
        lines=(output/(label+'.log')).read_text().splitlines()
        results=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('actual_dtype_audit ')];assert len(results)==1
        reports.append(results[0]);(output/'summary.json').write_text(json.dumps(dict(scope=metadata['scope'],reports=reports),indent=2)+'\n')
print('Completed six actual loaded-model dtype audits',flush=True)
