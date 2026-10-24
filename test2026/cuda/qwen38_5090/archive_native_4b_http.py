from pathlib import Path
import hashlib,json,subprocess,zipfile
r=Path('D:/CantorAI/5090-study-20261023');names=['native-4b-http-pilot','native-4b-http-quality-ledger','native-4b-http-benchmark-ledger','native-4b-http-server-ledger','native-4b-http-worker','native-4b-http-server-resources']
for name in ('native-4b-http-server-ledger','native-4b-http-quality-ledger','native-4b-http-benchmark-ledger'):
 events=[json.loads(l) for l in (r/'results'/name/'commands.jsonl').read_text(encoding='utf8').splitlines()];assert events[-1]['event']=='finished','Observe original process terminal before archival'
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
stop=r/'results/native-4b-http-intentional-stop.json';assert json.loads(stop.read_text(encoding='utf8'))['taskkill_exit']==0
files=[p for name in names for p in (r/'results'/name).rglob('*') if p.is_file()]+[stop]
files += [r/n for n in ['run-native-4b-http-pilot.py','run-native-4b-http-server.py','serve-native-4b-http-worker.bat','benchmark-native-4b-after-quality.py','stop-native-4b-http-worker.py','archive-native-4b-http.py','study-state.json','source-manifest.json','preparation-manifest.json','models/Qwen3.5-4B/config.json']]
p=r/'Garnet/test2026/cuda/qwen38_5090';files += [p/n for n in ['native_4b_stream_worker.py','native_http_bridge.py','native_worker_io.py','benchmark_stream.py','compare_http_runs.py','sample_vram.py','metrics.py','monitor_native_probe.py','run_logged.py']]
files += [r/'Garnet'/n for n in ['runtime/cuda_serving.py','xModel/qwen35/vl_0_8b/cuda_decoder.py']]
raw={p.relative_to(r).as_posix():p.read_bytes() for p in files};links={n:hashlib.sha256((r/'results'/n).read_bytes()).hexdigest() for n in ['qwen35-4b-grid16-evidence.zip','native-mtp3-shared-joined-evidence.zip']}
manifest={'scope':'Native4B BF16 B1/T16/cap4096 short-prompt HTTP study; raw quality/measurement/failures retained. Not quality parity with27B, GPU batching, tuned4B references or2x admission. Independent replay required.','library_sha256':hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest(),'linked_archives':links,'files':{n:hashlib.sha256(b).hexdigest() for n,b in raw.items()}}
out=r/'results/native-4b-http-evidence.zip';assert not out.exists()
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for n,b in raw.items():z.writestr(n,b)
 z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
print(json.dumps({'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'files':len(raw)}))
