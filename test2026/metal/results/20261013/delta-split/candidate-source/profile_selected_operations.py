"""Serialized diagnostic GPU attribution, never serving/benchmark timing.

Arguments: investigation root, new output directory, optional build directory
name, optional model name, optional batch. Run one actual native
workload at a time and require the unchanged complete baseline outputs.
"""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve();output.mkdir(parents=True,exist_ok=False)
scripts=root/'source/test2026/metal';build=root/(sys.argv[3] if len(sys.argv)>3 else 'build-vector-slice');runtime='/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3'
models=('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B')
if len(sys.argv)>4:
    assert sys.argv[4] in models;models=(sys.argv[4],)
batches=(1,2)
if len(sys.argv)>5:
    assert int(sys.argv[5]) in batches;batches=(int(sys.argv[5]),)
frontend=str(root/'comparators/vllm/venv/bin/python');statuses=[];summaries=[]

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
            if not identified or libraries.returncode or any(marker in libraries.stdout.lower() for marker in ('libgarnet','libmlx','libtorch','metal.framework','metalperformanceshaders.framework','libopenvino')):raise RuntimeError('Unverified native process PID '+str(pid))
        elif name.endswith(('/ollama','/vllm','/lossless_gpu_probe','/pruned_head_gpu','/prefill_tile_gpu')):raise RuntimeError('Other inference/GPU process PID '+str(pid))
    commands=subprocess.check_output(['/bin/ps','-axo','pid=,args='],text=True)
    for line in commands.splitlines():
        fields=line.strip().split(None,1)
        if len(fields)!=2 or int(fields[0])==os.getpid():continue
        if any(marker in fields[1] for marker in ('run_matched_vllm.py','vllm serve','vllm.entrypoints.openai','EngineCore::','frame_worker_vllm.py')):raise RuntimeError('Other inference worker PID '+fields[0])

metadata=dict(scope='Serialized per-operation attribution only; GPU/cache/dispatch behavior differs from serving',
    native_build=str(build),
    native_library_sha256=hashlib.sha256((build/'libgarnet.dylib').read_bytes()).hexdigest(),
    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),flag='GARNET_METAL_PROFILE_OPS=1')
(output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
for model in models:
    for batch in batches:
        label=model+'-b'+str(batch);folder=output/label;folder.mkdir();vl=model=='Qwen3.5-0.8B'
        baseline=root/'logs'/('vllm-matched-vl-prefill-wide-20261013' if vl else 'vllm-matched-text-prefill-wide-20261013')/'jobs'
        case=json.loads((baseline/(label+'-case.json')).read_text());case['requests']=1
        casefile=folder/'case.json';casefile.write_text(json.dumps(case,indent=2)+'\n')
        if vl:
            expected=json.loads((baseline/(label+'-garnet-0')/'protocol/command.json.event-0-frame_result.json').read_text())
            command=[frontend,str(scripts/'run_matched_frame_process.py'),runtime,str(build),str(root/'source'),str(root/'models'/model),str(casefile),'garnet',str(folder/'protocol')]
        else:
            text=(baseline/(label+'-garnet-0')/'output.log').read_text()
            expected=next(json.loads(line.split(' ',1)[1]) for line in text.splitlines() if line.startswith('matched_result '))
            command=[runtime,str(scripts/'run_matched_garnet.py'),str(build),str(root/'source'),str(root/'models'/model),str(casefile)]
        guard(label+'-before');begin=time.monotonic()
        with (folder/'output.log').open('w') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,GARNET_METAL_PROFILE_OPS='1'),start_new_session=True)
            try:code=process.wait(timeout=900)
            except subprocess.TimeoutExpired:
                assert os.getpgid(process.pid)==process.pid;os.killpg(process.pid,15)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,9);process.wait()
                code=process.returncode or 1
        status=dict(label=label,command=command,exit_code=code,diagnostic_wall_seconds=time.monotonic()-begin)
        statuses.append(status);(output/'statuses.json').write_text(json.dumps(statuses,indent=2)+'\n')
        if code:raise RuntimeError('Diagnostic failed: '+label)
        guard(label+'-after');text=(folder/'output.log').read_text()
        prefix='matched_frame_result ' if vl else 'matched_result '
        results=[json.loads(line.split(' ',1)[1]) for line in text.splitlines() if line.startswith(prefix)];assert len(results)==1
        result=results[0];assert result['token_ids']==expected['token_ids'] and result['prompt_token_ids']==expected['prompt_token_ids'] and result['finished']==expected['finished']
        log=(folder/'protocol/worker.log').read_text() if vl else text
        operations=[json.loads(line.split(' ',1)[1]) for line in log.splitlines() if line.startswith('metal_op_profile ')];assert operations
        grouped={}
        for operation in operations:
            assert operation['serialized_diagnostic']
            key=(operation['kind'],operation['a_rows'],operation['a_cols'],operation['out_rows'],operation['out_cols'],operation['bf16'])
            value=grouped.setdefault(key,dict(count=0,total_gpu_ms=0));value['count']+=1;value['total_gpu_ms']+=operation['gpu_ms']
        ranking=[dict(kind=key[0],a_rows=key[1],a_cols=key[2],out_rows=key[3],out_cols=key[4],bf16=key[5],**value) for key,value in grouped.items()]
        ranking.sort(key=lambda r:r['total_gpu_ms'],reverse=True)
        report=dict(model=model,batch=batch,exact_complete_native_output=True,scope=metadata['scope'],operations=len(operations),ranking=ranking)
        (folder/'attribution.json').write_text(json.dumps(report,indent=2)+'\n');summaries.append(report)
        (output/'summary.json').write_text(json.dumps(dict(scope=metadata['scope'],profiles=summaries),indent=2)+'\n')
        print('Completed diagnostic',label,'operations',len(operations),'top',ranking[:3],flush=True)
