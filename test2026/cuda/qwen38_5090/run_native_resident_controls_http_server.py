from pathlib import Path
import hashlib,json,socket,subprocess,sys
r=Path('D:/CantorAI/5090-study-20261023');p=r/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(p));from run_logged import run
proof=json.loads((r/'results/warm-mtp-resident-controls-attempt1-verified.json').read_text());assert proof['status']=='resident_controls_correctness_evidence_verified'
assert proof['archive_sha256']==hashlib.sha256((r/'results/warm-mtp-resident-controls-attempt1-evidence.zip').read_bytes()).hexdigest()
assert hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest()=='ba67dc9d4f8b9948615e522e3b6ad0aa10ea6f742505465084449fe57ead0b8b'
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024,'Inspect current GPU owner before starting resident-controls27B'
with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',18082))!=0,'Inspect occupied port; never replace unrelated owner'
ledger=r/'results/native-resident-controls-http-server-ledger';worker=r/'results/native-resident-controls-http-worker';resources=r/'results/native-resident-controls-http-server-resources';assert all(not d.exists() for d in (ledger,worker,resources))
launcher=r/'serve-native-resident-controls-worker.bat'
command=[sys.executable,str(p/'monitor_native_probe.py'),'--study',str(r),'--output',str(resources),'--',sys.executable,str(p/'native_http_bridge.py'),'--model','qwen38-27b-nvfp4','--port','18082','--log-root',str(worker),'--','cmd.exe','/d','/c',str(launcher)]
raise SystemExit(run(command,ledger,'correctness','native-resident-controls-http-worker',[launcher,r/'build/bin/garnet.dll',r/'build/bin/xlang3.exe',p/'native_stream_worker.py',p/'native_worker_profile.py']))
