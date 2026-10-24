from pathlib import Path
import hashlib,json,sys,time
r=Path('D:/CantorAI/5090-study-20261023');ledger=r/'results/native-trt-layer-profiler-fixed-preflight-ledger/commands.jsonl'
print('WAITING_FOR_BOUNDED_PROFILER_QUALIFICATION',flush=True)
while True:
 if ledger.exists():
  events=[json.loads(line) for line in ledger.read_text().splitlines()]
  if events and events[-1]['event']=='finished':
   assert events[-1]['exit_code']==0,'Bounded qualification failed; full model not launched'
   break
 time.sleep(5)
report=json.loads((r/'results/native-trt-layer-profiler-fixed-preflight/report.json').read_text())
assert report['status']=='passed'
assert report['library_sha256']==hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest()
sys.path.insert(0,str(r/'Garnet/test2026/cuda/qwen38_5090'))
from run_logged import run
command=['C:/Python312/python.exe',str(r/'monitor_native_probe.py'),'--study',str(r),'--output',str(r/'results/shared-trt-layer-diagnostic-resources'),'--','cmd.exe','/d','/c',str(r/'test-shared-trt-layers.bat')]
raise SystemExit(run(command,r/'results/shared-trt-layer-diagnostic-ledger','correctness','first-use-eager-layer-diagnostic',[r/'test-shared-trt-layers.bat',r/'Garnet/test2026/cuda/qwen38_5090/probe_shared_trt_layers.py']))
