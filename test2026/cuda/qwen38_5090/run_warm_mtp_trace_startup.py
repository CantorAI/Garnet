from pathlib import Path
import hashlib,json,shutil,subprocess,sys
r=Path('D:/CantorAI/5090-study-20261023');p=r/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(p));from run_logged import run
failed=r/'results/warm-mtp-cuda-trace-controller-ledger/commands.jsonl';events=[json.loads(l) for l in failed.read_text(encoding='utf8').splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==1
assert (r/'results/warm-mtp-cuda-trace-attempt1-evidence.zip').exists(),'Archive original failure before new experiment'
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
assert shutil.disk_usage(r).free>5*(1<<30),'Inspect available disk before full startup trace'
assert hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest()=='ba67dc9d4f8b9948615e522e3b6ad0aa10ea6f742505465084449fe57ead0b8b'
out=r/'results/warm-mtp-cuda-trace-startup-attempt1';out.mkdir(exist_ok=False);nsys=Path('C:/Program Files/NVIDIA Corporation/Nsight Systems 2024.6.2/target-windows-x64/nsys.exe');launcher=r/'test-warm-mtp-trace-startup.bat'
report={'status':'launching','scope':'Collect from process launch; select warm window only after validating unique HtoD markers in raw SQLite. Full report contains cold/build activity and is not warm performance. No HTTP or2x claim.'}
def save():(out/'workflow.json').write_text(json.dumps(report,indent=2),encoding='utf8')
save()
code=run([str(nsys),'profile','--trace=cuda','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--force-overwrite=false','--output='+str(out/'startup-trace'),'cmd.exe','/d','/c',str(launcher)],out/'profile-ledger','correctness','startup-instrumented-warm-mtp',[nsys,launcher,r/'probe-warm-mtp-trace-startup.py',r/'build/bin/garnet.dll'])
report['profile_exit_code']=code;report['status']='profile_failed' if code else 'profile_finished_unverified';save()
if code:raise SystemExit(1)
assert json.loads((out/'native/report.json').read_text())['status']=='passed'
code=run([str(nsys),'stats','--report','cuda_gpu_kern_sum,cuda_api_sum','--format','csv',str(out/'startup-trace.nsys-rep')],out/'stats-ledger','correctness','startup-trace-raw-stats-not-warm-ranking',[nsys])
report['stats_exit_code']=code;report['status']='native_exact_marked_trace_unverified';save();print(json.dumps(report));raise SystemExit(code)
