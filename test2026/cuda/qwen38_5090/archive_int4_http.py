from pathlib import Path
import json,hashlib,zipfile
r=Path('D:/CantorAI/5090-study-20261023');names=['native-int4-http-pilot','native-int4-worker','native-int4-server-ledger','native-int4-http-workflow-ledger']
for name in names:
 p=r/'results'/name/'commands.jsonl'
 if p.exists():assert json.loads(p.read_text().splitlines()[-1])['event']=='finished','Wait for terminal handle'
stop=json.loads((r/'results/native-int4-intentional-stop.json').read_text());assert stop['taskkill_exit']==0
pilot=r/'results/native-int4-http-pilot';assert json.loads((pilot/'completed.json').read_text())['status']=='completed'
paths=list((pilot/'benchmark').glob('*-manifest.json'));assert len(paths)==1
m=json.loads(paths[0].read_text());assert m['status']=='completed'
files=[p for name in names for p in (r/'results'/name).rglob('*') if p.is_file()]
files += [r/x for x in ['stop-int4-worker.py','archive-int4-http.py','results/native-int4-intentional-stop.json','results/int4-http-source-install.json','results/int4-packed-qualification/report.json','study-state.json','preparation-manifest.json','int4-http-source.zip','models/Qwen3.8-27B-INT4/config.json','models/Qwen3.8-27B-INT4/model.safetensors.index.json']]
p=r/'Garnet/test2026/cuda/qwen38_5090'
files += [p/x for x in ['native_int4_stream_worker.py','native_http_bridge.py','serve_native_int4_worker.bat','run_int4_http.py','benchmark_stream.py','metrics.py','sample_vram.py']]
for row in json.loads((r/'results/int4-packed-qualification/report.json').read_text())['cases']:
 files.append(r/'results/native-mtp3-http'/(row['key']+'-request.json'))
raw={p.relative_to(r).as_posix():p.read_bytes() for p in files};out=r/'results/native-int4-http-evidence.zip';assert not out.exists()
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for n,b in raw.items():z.writestr(n,b)
 z.writestr('evidence-manifest.json',json.dumps({'files':{n:hashlib.sha256(b).hexdigest() for n,b in raw.items()},'binary_sha256':hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest(),'scope':'Single INT4 B1 short-prompt pilot; compare precisions cautiously. No tuned matched INT4 references, broad quality or 2x claim.'},indent=2))
print(json.dumps({'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'entries':len(raw)+1,'cohorts':m['cohorts']}))
