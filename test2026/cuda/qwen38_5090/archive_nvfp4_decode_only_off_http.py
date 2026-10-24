from pathlib import Path
import hashlib,json,zipfile
r=Path('D:/CantorAI/5090-study-20261023')
names=['native-mtp3-decode-only-off-workflow-ledger','native-mtp3-decode-only-off-server-ledger','native-mtp3-decode-only-off-workflow','native-mtp3-decode-only-off-worker','native-mtp3-decode-only-off-server-resources','shared-decode-only-off-http-diagnostic']
terminal={}
for name in names[:2]:
 events=[json.loads(line) for line in (r/'results'/name/'commands.jsonl').read_text(encoding='utf8').splitlines()]
 assert events[-1]['event']=='finished','Observe original job terminal before archiving'
 terminal[name]=events[-1]
stop=r/'results/native-mtp3-decode-only-off-intentional-stop.json'
assert json.loads(stop.read_text(encoding='utf8'))['taskkill_exit']==0
files=[p for name in names for p in (r/'results'/name).rglob('*') if p.is_file()]
root_names=['run-nvfp4-decode-only-off-server.py','run-nvfp4-decode-only-off-workflow.py','serve-nvfp4-decode-only-off-worker.bat','stop-nvfp4-decode-only-off-worker.py','archive-nvfp4-decode-only-off-http.py']
files += [r/n for n in root_names]+[stop,r/'results/nvfp4-decode-only-full-independent-verification.json']
p=r/'Garnet/test2026/cuda/qwen38_5090'
files += [p/n for n in ['native_http_bridge.py','native_stream_worker.py','monitor_native_probe.py','benchmark_stream.py','probe_shared_mtp_sequences.py','run_logged.py','joined_model_gates.py','validate_native_mtp_http.py','compare_http_runs.py','sample_vram.py','metrics.py','native_worker_profile.py','native_worker_io.py','native_prompt.py','run_shared_mtp_candidate.py']]
raws={f.relative_to(r).as_posix():f.read_bytes() for f in files}
anchors={n:hashlib.sha256((r/'results'/n).read_bytes()).hexdigest() for n in ['nvfp4-decode-only-full-evidence.zip','native-mtp3-decode-only-http-evidence.zip','native-mtp3-shared-joined-evidence.zip']}
manifest={'scope':'Untimed same-DLL V3 option-off HTTP20 control. Preserve failure or success; independent replay required. No speed or batching qualification.','terminal':terminal,'linked_archives':anchors,'files':{n:hashlib.sha256(b).hexdigest() for n,b in raws.items()},'library_sha256':hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest()}
out=r/'results/native-mtp3-decode-only-off-http-evidence.zip';assert not out.exists()
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for n,b in raws.items():z.writestr(n,b)
 z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
print(json.dumps({'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'files':len(raws),'terminal':terminal}))
