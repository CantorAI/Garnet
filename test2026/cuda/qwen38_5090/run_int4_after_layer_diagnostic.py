from pathlib import Path
import hashlib,json,subprocess,sys,time
r=Path('D:/CantorAI/5090-study-20261023');ledger=r/'results/shared-trt-layer-diagnostic-ledger/commands.jsonl'
print('WAITING_FOR_ORIGINAL_FULL_LAYER_DIAGNOSTIC',flush=True)
while True:
 events=[json.loads(line) for line in ledger.read_text().splitlines()]
 if events and events[-1]['event']=='finished':break
 time.sleep(5)
# Preserve the diagnostic, including any failure, before proceeding.
subprocess.run(['C:/Python312/python.exe',str(r/'archive-full-trt-layers.py')],check=True)
assert events[-1]['exit_code']==0,'Full layer diagnostic failed; investigate before further GPU work'
report=json.loads((r/'results/shared-trt-layer-diagnostic/report.json').read_text())
assert report['status']=='passed'
assert report['library_sha256']==hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest()
sys.path.insert(0,str(r/'Garnet/test2026/cuda/qwen38_5090'))
from run_logged import run
command=['C:/Python312/python.exe',str(r/'monitor_native_probe.py'),'--study',str(r),'--output',str(r/'results/int4-packed-qualification-resources'),'--','cmd.exe','/d','/c',str(r/'test-int4-packed-full.bat')]
raise SystemExit(run(command,r/'results/int4-packed-qualification-ledger','correctness','int4-packed-versus-perlayer-golden',[r/'test-int4-packed-full.bat',r/'Garnet/test2026/cuda/qwen38_5090/probe_int4_packed_full.py']))
