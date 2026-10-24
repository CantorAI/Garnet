from pathlib import Path
import hashlib,json,socket,sys
r=Path('D:/CantorAI/5090-study-20261023');p=r/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(p));from run_logged import run
proof=r/'results/nvfp4-decode-only-full-independent-verification.json';verified=json.loads(proof.read_text(encoding='utf8'))
assert verified['status']=='verified' and verified['archive_sha256']==hashlib.sha256((r/'results/nvfp4-decode-only-full-evidence.zip').read_bytes()).hexdigest()
dll=r/'build/bin/garnet.dll';assert verified['library_sha256']==hashlib.sha256(dll.read_bytes()).hexdigest()
with socket.socket() as sock:sock.settimeout(2);assert sock.connect_ex(('127.0.0.1',18082))!=0,'Occupied study port; inspect owner, do not replace it'
ledger=r/'results/native-mtp3-decode-only-off-server-ledger';worker=r/'results/native-mtp3-decode-only-off-worker';assert not ledger.exists() and not worker.exists()
launcher=r/'serve-nvfp4-decode-only-off-worker.bat'
resources=r/'results/native-mtp3-decode-only-off-server-resources';assert not resources.exists()
command=[sys.executable,str(p/'monitor_native_probe.py'),'--study',str(r),'--output',str(resources),'--',sys.executable,str(p/'native_http_bridge.py'),'--port','18082','--log-root',str(worker),'--','cmd.exe','/d','/c',str(launcher)]
code=run(command,ledger,'correctness','v3-option-off-control-worker',[launcher,dll,r/'build/bin/xlang3.exe',proof,p/'monitor_native_probe.py'])
raise SystemExit(code)
