from pathlib import Path
import hashlib,json,socket,subprocess,sys
r=Path('D:/CantorAI/5090-study-20261023');p=r/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(p));from run_logged import run
# Launch only after the original option-off worker has really closed.
events=[json.loads(l) for l in (r/'results/native-mtp3-decode-only-off-server-ledger/commands.jsonl').read_text(encoding='utf8').splitlines()];assert events[-1]['event']=='finished'
assert json.loads((r/'results/native-mtp3-decode-only-off-intentional-stop.json').read_text(encoding='utf8'))['taskkill_exit']==0
with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',18082))!=0
used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True);assert int(used.strip())<1024,'Inspect GPU owner before profiling'
out=r/'results/nsight-cuda-smoke-attempt1';out.mkdir(exist_ok=False)
nsys=Path('C:/Program Files/NVIDIA Corporation/Nsight Systems 2024.6.2/target-windows-x64/nsys.exe');launcher=r/'test-nsight-cuda-smoke.bat';version=subprocess.check_output([str(nsys),'--version'],text=True)
(out/'identity.json').write_text(json.dumps({'nsys_version':version,'nsys_sha256':hashlib.sha256(nsys.read_bytes()).hexdigest(),'library_sha256':hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest(),'scope':'Capability smoke only: real bounded graph, not model throughput. A successful exit alone does not prove CUDA trace coverage.'},indent=2),encoding='utf8')
cmd=[str(nsys),'profile','--trace=cuda','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--force-overwrite=false','--output='+str(out/'normalization-trace'),'cmd.exe','/d','/c',str(launcher)]
code=run(cmd,out/'profile-ledger','correctness','nsight-cuda-capability-smoke',[nsys,launcher,r/'build/bin/garnet.dll',p/'probe_trt_layer_profiler.py'])
report={'profile_exit_code':code,'status':'profile_failed' if code else 'profile_finished_trace_unverified','scope':'No performance claim; inspect actual GPU kernel/API records before declaring profiler usable.'}
if code==0:
 stats=run([str(nsys),'stats','--report','cuda_gpu_kern_sum,cuda_api_sum','--format','csv',str(out/'normalization-trace.nsys-rep')],out/'stats-ledger','correctness','nsight-smoke-raw-stats',[nsys])
 report['stats_exit_code']=stats
(out/'workflow.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report))
raise SystemExit(code)
