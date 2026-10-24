from pathlib import Path
import hashlib,json,socket,sys
r=Path('D:/CantorAI/5090-study-20261023');p=r/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(p));from run_logged import run
assert hashlib.sha256((r/'models/Qwen3.5-4B/config.json').read_bytes()).hexdigest()=='ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670'
assert hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest()=='ba67dc9d4f8b9948615e522e3b6ad0aa10ea6f742505465084449fe57ead0b8b'
with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',18082))!=0,'Inspect occupied port; never replace unrelated owner'
ledger=r/'results/native-4b-http-server-ledger';worker=r/'results/native-4b-http-worker';resources=r/'results/native-4b-http-server-resources';assert all(not d.exists() for d in (ledger,worker,resources))
launcher=r/'serve-native-4b-http-worker.bat'
command=[sys.executable,str(p/'monitor_native_probe.py'),'--study',str(r),'--output',str(resources),'--',sys.executable,str(p/'native_http_bridge.py'),'--model','qwen35-4b-bf16','--port','18082','--log-root',str(worker),'--','cmd.exe','/d','/c',str(launcher)]
raise SystemExit(run(command,ledger,'correctness','native-4b-http-worker',[launcher,r/'build/bin/garnet.dll',r/'build/bin/xlang3.exe',p/'native_4b_stream_worker.py']))
