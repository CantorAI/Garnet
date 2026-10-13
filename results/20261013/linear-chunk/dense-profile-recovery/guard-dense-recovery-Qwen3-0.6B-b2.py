import json,os,subprocess
from pathlib import Path
output=Path('/Users/zsdx/garnet-metal-investigation/logs/dense-recovery-preflight-Qwen3-0.6B-b2-20261013');output.mkdir(parents=True,exist_ok=False)
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
        elif name.endswith(('/ollama','/vllm','/lossless_gpu_probe','/pruned_head_gpu','/prefill_tile_gpu')):raise RuntimeError('Other inference/GPU process PID '+str(pid))
    commands=subprocess.check_output(['/bin/ps','-axo','pid=,args='],text=True)
    for line in commands.splitlines():
        fields=line.strip().split(None,1)
        if len(fields)!=2 or int(fields[0])==os.getpid():continue
        if any(marker in fields[1] for marker in ('run_matched_vllm.py','vllm serve','vllm.entrypoints.openai','EngineCore::','frame_worker_vllm.py')):
            raise RuntimeError('Other inference worker PID '+fields[0])

guard('before-build');print('Current Mac inference exclusivity guard passed')
