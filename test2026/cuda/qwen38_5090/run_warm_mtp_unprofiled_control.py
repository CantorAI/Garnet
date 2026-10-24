from pathlib import Path
import hashlib,json,subprocess,sys
r=Path('D:/CantorAI/5090-study-20261023');p=r/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(p));from run_logged import run
assert (r/'results/warm-mtp-cuda-trace-startup-attempt1-evidence.zip').exists()
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
out=r/'results/warm-mtp-trace-unprofiled-control';out.mkdir(exist_ok=False);launcher=r/'test-warm-mtp-unprofiled-control.bat';assert hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest()=='ba67dc9d4f8b9948615e522e3b6ad0aa10ea6f742505465084449fe57ead0b8b'
code=run([sys.executable,str(p/'monitor_native_probe.py'),'--study',str(r),'--output',str(out/'resources'),'--','cmd.exe','/d','/c',str(launcher)],out/'ledger','correctness','identical-probe-without-nsys',[launcher,r/'probe-warm-mtp-trace-startup.py',r/'build/bin/garnet.dll'])
report={'status':'process_failed' if code else 'process_finished_needs_independent_verification','exit_code':code,'scope':'Identical14-case startup-marker probe without Nsight; untimed control. Not a profiler cause or model-speed claim.'}
if code==0:assert json.loads((out/'native/report.json').read_text())['status']=='passed'
(out/'workflow.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report));raise SystemExit(0 if code==0 else 1)
