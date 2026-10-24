from pathlib import Path
import json,subprocess,sys,time
r=Path('D:/CantorAI/5090-study-20261023');ledger=r/'results/profiler-lifecycle-build-ledger/commands.jsonl'
print('WAITING_FOR_ORIGINAL_BUILD_LEDGER',flush=True)
while True:
 events=[json.loads(line) for line in ledger.read_text().splitlines()]
 latest=events[-1]
 if latest['event']=='finished':
  assert latest['exit_code']==0,'Original build failed; no correctness process launched'
  assert latest['label']=='profiler-lifecycle-fixed-launcher'
  break
 time.sleep(5)
sys.path.insert(0,str(r/'Garnet/test2026/cuda/qwen38_5090'))
from run_logged import run
command=['C:/Python312/python.exe',str(r/'monitor_native_probe.py'),'--study',str(r),'--output',str(r/'results/native-trt-layer-profiler-fixed-preflight-resources'),'--','cmd.exe','/d','/c',str(r/'test-trt-layer-profiler-fixed.bat')]
code=run(command,r/'results/native-trt-layer-profiler-fixed-preflight-ledger','correctness','profiler-lifecycle-bounded-preflight',[r/'profiler-lifecycle-source.zip',r/'test-trt-layer-profiler-fixed.bat'])
if code:raise SystemExit(code)
assert json.loads((r/'results/native-trt-layer-profiler-fixed-preflight/report.json').read_text())['status']=='passed'
subprocess.run(['C:/Python312/python.exe',str(r/'archive-profiler-fixed.py')],check=True)
print('PROFILER_LIFECYCLE_QUALIFIED_AND_ARCHIVED',flush=True)
