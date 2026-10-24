from pathlib import Path
import hashlib,json,subprocess,zipfile
r=Path('D:/CantorAI/5090-study-20261023');names=['layer0-real-stage-attempt2','layer0-real-stage-attempt2-controller-ledger','layer0-stage-probe-fix-install','layer0-stage-probe-fix-install-ledger'];events=[json.loads(l) for l in (r/'results'/names[1]/'commands.jsonl').read_text().splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
report=json.loads((r/'results/layer0-real-stage-attempt2/native/report.json').read_text());assert report['status']=='witness_passed_needs_independent_array_replay' and len(report['files'])==81
files=[p for n in names for p in (r/'results'/n).rglob('*') if p.is_file()];files += [r/n for n in ['run-layer0-real-stages-warmed.py','test-layer0-real-stages-warmed.bat','install-layer0-stage-probe-fix.py','probe-layer0-real-stages-warmed.py','archive-layer0-stage-attempt2.py','results/layer0-stage-witness/manifest.json','results/layer0-stage-build/manifest.json']]
files += [r/'Garnet'/n for n in ['test2026/cuda/qwen38_5090/probe_layer0_real_stages.py','xModel/qwen35/vl_0_8b/cuda_packed_decoder.py','src/backends/tensorrt/qwen35_lowering.h','src/backends/tensorrt/trt_builder.cpp','test2026/cuda/qwen38_5090/run_logged.py','test2026/cuda/qwen38_5090/monitor_native_probe.py']]
raw={p.relative_to(r).as_posix():p.read_bytes() for p in files};library=hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest();assert library==report['library_sha256']
links={n:hashlib.sha256((r/'results'/n).read_bytes()).hexdigest() for n in ['layer0-real-stage-attempt1-evidence.zip','native-prefix12-internals-evidence.zip','native-controls-abba-evidence.zip']}
m={'scope':'Corrected actual-input/fenced-warmup layer0 B1/B2 stage witness. Full primary/repeat/row/array replay required before localizing first observable stage; no full64 batching, unique rootcause, serving or speed admission.','terminal':events[-1],'library_sha256':library,'linked_archives':links,'files':{n:hashlib.sha256(b).hexdigest() for n,b in raw.items()}}
out=r/'results/layer0-real-stage-attempt2-evidence.zip';assert not out.exists()
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for n,b in raw.items():z.writestr(n,b)
 z.writestr('evidence-manifest.json',json.dumps(m,indent=2))
print(json.dumps({'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'files':len(raw)}))
