from pathlib import Path
import hashlib,json,subprocess,zipfile
r=Path('D:/CantorAI/5090-study-20261023');names=['native-resident-controls-http-controller-ledger','native-resident-controls-http-server-ledger','native-resident-controls-http-workflow','native-resident-controls-http-worker','native-resident-controls-http-server-resources','shared-resident-controls-http-diagnostic','shared-resident-controls-http-diagnostic-ledger','native-resident-controls-profile-pilot','resident-controls-http-source-install','resident-controls-http-source-install-ledger']
terminal={}
for n in names[:2]:
 events=[json.loads(l) for l in (r/'results'/n/'commands.jsonl').read_text(encoding='utf8').splitlines()];assert events[-1]['event']=='finished','Observe original handle terminal first';terminal[n]=events[-1]
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
stop=r/'results/native-resident-controls-http-intentional-stop.json';assert json.loads(stop.read_text())['taskkill_exit']==0
files=[p for n in names for p in (r/'results'/n).rglob('*') if p.is_file()]+[stop]
files += [r/n for n in ['run-native-resident-controls-http-server.py','run-native-resident-controls-http-workflow.py','run-native-resident-controls-http-candidate.py','serve-native-resident-controls-worker.bat','stop-native-resident-controls-http-worker.py','archive-native-resident-controls-http.py','install-resident-controls-http.py','resident-controls-http-install.zip','results/warm-mtp-resident-controls-attempt1-verified.json','results/shared-decoder-full-qualification/report.json','models/Qwen3.8-27B-NVFP4/config.json']]
p=r/'Garnet/test2026/cuda/qwen38_5090';files += [p/n for n in ['native_http_bridge.py','native_stream_worker.py','native_worker_profile.py','monitor_native_probe.py','benchmark_stream.py','probe_shared_mtp_sequences.py','run_logged.py','validate_native_mtp_http.py','compare_http_runs.py','sample_vram.py','metrics.py','native_worker_io.py','native_prompt.py']]
files += [r/'Garnet'/n for n in ['xModel/qwen35/vl_0_8b/cuda_speculative.py','xModel/qwen35/vl_0_8b/cuda_packed_replay.py','xModel/qwen35/vl_0_8b/cuda_hybrid.py','xModel/qwen35/vl_0_8b/cuda_verify.py','src/backends/tensorrt/cuda_tensor_io.cpp','src/cuda/qwen35_io.cu']]
raw={f.relative_to(r).as_posix():f.read_bytes() for f in files};links={n:hashlib.sha256((r/'results'/n).read_bytes()).hexdigest() for n in ['warm-mtp-resident-controls-attempt1-evidence.zip','native-mtp3-shared-joined-evidence.zip','native-mtp3-decode-only-off-http-evidence.zip','warm-mtp-cuda-trace-2026-attempt1-evidence.zip']}
e={'scope':'Resident request-control native27B joined MTP3 B1 pilot including failures. Independent full-output/SSE/VRAM reaccounting required. No same-binary repeated A/B, true GPU batching, broad quality/fullcontext or2x admission.','terminal':terminal,'library_sha256':hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest(),'runtime_sha256':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ['build/bin/garnet.dll','build/bin/xlang3.exe']},'linked_archives':links,'files':{n:hashlib.sha256(b).hexdigest() for n,b in raw.items()}}
out=r/'results/native-resident-controls-http-evidence.zip';assert not out.exists()
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for n,b in raw.items():z.writestr(n,b)
 z.writestr('evidence-manifest.json',json.dumps(e,indent=2))
print(json.dumps({'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'files':len(raw),'terminal':terminal}))
