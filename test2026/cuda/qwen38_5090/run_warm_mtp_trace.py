from pathlib import Path
import concurrent.futures,hashlib,json,subprocess,sys,time
r=Path('D:/CantorAI/5090-study-20261023');p=r/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(p));from run_logged import run
smoke=r/'results/nsight-cuda-smoke-attempt1/workflow.json';assert json.loads(smoke.read_text())['stats_exit_code']==0
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024,'Inspect GPU owner, never overlap models'
out=r/'results/warm-mtp-cuda-trace-attempt1';out.mkdir(exist_ok=False);session='garnet5090warm1';nsys=Path('C:/Program Files/NVIDIA Corporation/Nsight Systems 2024.6.2/target-windows-x64/nsys.exe');launcher=r/'test-warm-mtp-trace.bat'
assert hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest()=='ba67dc9d4f8b9948615e522e3b6ad0aa10ea6f742505465084449fe57ead0b8b'
report={'status':'launching','session':session,'scope':'Separate warm joined-MTP CUDA node trace; profiler overhead included. No HTTP, uninstrumented speed, concurrency, broad-quality or2x claim.'}
def save(): (out/'workflow.json').write_text(json.dumps(report,indent=2),encoding='utf8')
save()
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
 future=pool.submit(run,[str(nsys),'launch','--session-new='+session,'--wait=all','--trace=cuda','--cuda-graph-trace=node','cmd.exe','/d','/c',str(launcher)],out/'launch-ledger','correctness','warm-mtp-trace-launch',[nsys,launcher,r/'probe-warm-mtp-trace.py',r/'build/bin/garnet.dll'])
 def wait_file(name):
  deadline=time.monotonic()+1800
  while not (out/name).exists():
   if future.done():raise RuntimeError('Original launch terminal before '+name+'; exit='+str(future.result()))
   if time.monotonic()>=deadline:raise TimeoutError('Inspect original launch handle; no restart on this deadline')
   time.sleep(.2)
 try:
  wait_file('warm-ready.json');report['status']='starting_capture';save()
  code=run([str(nsys),'start','--session='+session,'--sample=none','--cpuctxsw=none','--force-overwrite=false','--output='+str(out/'warm-trace')],out/'start-ledger','correctness','warm-mtp-trace-start',[nsys]);assert code==0
  (out/'capture-go.json').write_text('{}');report['status']='capturing';save();wait_file('capture-done.json')
  code=run([str(nsys),'stop','--session='+session],out/'stop-ledger','correctness','warm-mtp-trace-stop',[nsys]);assert code==0
 finally:
  (out/'abort.json').write_text('{}');(out/'release.json').write_text('{}')
 code=future.result();assert code==0
assert json.loads((out/'native/report.json').read_text())['status']=='passed'
code=run([str(nsys),'stats','--report','cuda_gpu_kern_sum,cuda_api_sum','--format','csv',str(out/'warm-trace.nsys-rep')],out/'stats-ledger','correctness','warm-mtp-raw-trace-stats',[nsys])
report.update(status='native_exact_trace_unverified',stats_exit_code=code);save();print(json.dumps(report));raise SystemExit(code)
