"""Sequential native ABBA prefill profile experiment with exact output checks."""
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
build=root/'build-linear-chunk';control=root/'build-parallel-delta'
metadata=dict(scope='Native ABBA batched recurrent output projection and adaptive tails; no vLLM speed claim',candidate_sha256=digest(build/'libgarnet.dylib'),control_sha256=digest(control/'libgarnet.dylib'),worker_sha256=digest(scripts/'frame_worker_garnet.py'),decoder_sha256=digest(root/'source/xModel/qwen35/vl_0_8b/metal_decoder.py'))
(output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
records=[];groups=[]
for batch in (1,2):
 label='Qwen3.5-0.8B-b'+str(batch);baseline=root/'logs/native-frame-ab-parallel-delta-confirm-20261013/jobs'
 original=json.loads((baseline/(label+'-case.json')).read_text());expected=json.loads((baseline/(label+'-garnet-0')/'protocol/command.json.event-0-frame_result.json').read_text())
 for chunk in (16,32,64):
  for repeat in range(2):
   for backend in (('candidate','control') if repeat==0 else ('control','candidate')):
    name=label+'-c'+str(chunk)+'-'+backend+'-'+str(repeat);folder=output/name;folder.mkdir()
    case=dict(original);case['chunk']=chunk if backend=='candidate' else 16
    case['prefill_linear_chunk']=backend=='candidate'
    casefile=folder/'case.json';casefile.write_text(json.dumps(case,indent=2)+'\n');guard(name+'-before')
    selected=build if backend=='candidate' else control
    run(name,[sys.executable,str(scripts/'benchmark_matched_process.py'),frontend,str(scripts/'run_matched_frame_process.py'),runtime,str(selected),str(root/'source'),str(root/'models/Qwen3.5-0.8B'),str(casefile),'garnet',str(folder/'protocol')])
    guard(name+'-after');lines=(output/(name+'.log')).read_text().splitlines()
    values=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('matched_frame_result ')];assert len(values)==3
    memory=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('matched_process ')];assert len(memory)==1 and memory[0]['exit_code']==0
    for value in values:
     assert value['token_ids']==expected['token_ids'] and value['finished']==expected['finished'] and value['prompt_token_ids']==expected['prompt_token_ids']
     plan=([16,16,16,16,16] if chunk==16 else [32,32,16] if chunk==32 else [64,16]) if backend=='candidate' else [16,16,16,16,16]
     assert value['prefill_chunk_plan']==plan
     value.update(repeat=repeat,backend=backend,chunk_profile=chunk,process=memory[0]);records.append(value)
  profile=[v for v in records if v['case']['batch']==batch and v['chunk_profile']==chunk]
  def mean(backend):
   values=[v for v in profile if v['backend']==backend and v['request']>0]
   return {key:statistics.mean(v[key] for v in values) for key in ('ttft_seconds','decode_wall_steps_per_second','total_seconds','prefill_wall_seconds')}
  candidate=mean('candidate');reference=mean('control')
  group=dict(batch=batch,chunk=chunk,candidate=candidate,control=reference,ttft_fraction=candidate['ttft_seconds']/reference['ttft_seconds'],decode_ratio=candidate['decode_wall_steps_per_second']/reference['decode_wall_steps_per_second'],exact_complete_outputs=True,
   first_use={b:statistics.mean(v['ttft_seconds'] for v in profile if v['backend']==b and v['request']==0) for b in ('candidate','control')},
   peak_individual_child_rss_bytes={b:max(v['process']['peak_rss_bytes'] for v in profile if v['backend']==b) for b in ('candidate','control')})
  groups.append(group);(output/'summary.json').write_text(json.dumps(dict(scope=metadata['scope'],groups=groups,records=records),indent=2)+'\n');print(json.dumps(group),flush=True)
