from pathlib import Path
import hashlib,json,subprocess,sys
r=Path('D:/CantorAI/5090-study-20261023');p=r/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(p));from run_logged import run
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
assert hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest()=='ba67dc9d4f8b9948615e522e3b6ad0aa10ea6f742505465084449fe57ead0b8b'
events=[json.loads(l) for l in (r/'results/warm-mtp-cuda-trace-2026-controller-ledger/commands.jsonl').read_text().splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
verified=json.loads((r/'results/warm-mtp-cuda-trace-2026-attempt1-verified.json').read_text());assert verified['status']=='warm_trace_evidence_verified'
out=r/'results/warm-mtp-resident-controls-attempt1';out.mkdir(exist_ok=False)
for name,launcher in [('transport','test-resident-controls-transport.bat'),('model','test-warm-mtp-resident-controls.bat')]:
 code=run([sys.executable,str(p/'monitor_native_probe.py'),'--study',str(r),'--output',str(out/(name+'-resources')),'--','cmd.exe','/d','/c',str(r/launcher)],out/(name+'-ledger'),'correctness','resident-controls-'+name,[r/launcher,r/'probe-warm-mtp-resident-controls.py',r/'build/bin/garnet.dll',r/'Garnet/test2026/cuda/test_qwen35_resident_controls.py'])
 if code:raise SystemExit(code)
 if name=='transport':assert json.loads((out/'transport.json').read_text())['all_passed']
 else:
  report=json.loads((out/'native/report.json').read_text());assert report['status']=='passed' and report['resident_controls'] and len(report['cases'])==14
(out/'workflow.json').write_text(json.dumps({'status':'producer_correctness_passed_needs_independent_verification','scope':'Transport48 trials and14 full-model cases; no HTTP or performance admission.'},indent=2))
