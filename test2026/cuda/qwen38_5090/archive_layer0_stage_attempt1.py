from pathlib import Path
import hashlib,json,subprocess,zipfile
r=Path('D:/CantorAI/5090-study-20261023');names=['layer0-real-stage-attempt1','layer0-real-stage-controller-ledger','layer0-stage-source-install','layer0-stage-source-install-ledger','layer0-stage-build','layer0-stage-build-controller-ledger'];terminal={}
for n,expected in [('layer0-real-stage-controller-ledger',1),('layer0-stage-source-install-ledger',0),('layer0-stage-build-controller-ledger',0)]:
 events=[json.loads(l) for l in (r/'results'/n/'commands.jsonl').read_text().splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==expected;terminal[n]=events[-1]
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
report=json.loads((r/'results/layer0-real-stage-attempt1/native/report.json').read_text());assert report['status']=='failed' and len(report['files'])==26
files=[p for n in names for p in (r/'results'/n).rglob('*') if p.is_file()]
files += [r/n for n in ['run-layer0-real-stages.py','test-layer0-real-stages.bat','run-layer0-stage-build.py','build-layer0-stages.bat','install-layer0-stage-source.py','layer0-stage-source.zip','archive-layer0-stage-attempt1.py','results/layer0-stage-witness/manifest.json']]
files += [r/'Garnet'/n for n in ['src/backends/tensorrt/qwen35_lowering.h','src/backends/tensorrt/trt_builder.cpp','test2026/cuda/qwen38_5090/probe_layer0_real_stages.py','test2026/cuda/qwen38_5090/prepare_layer0_stage_witness.py','xModel/qwen35/vl_0_8b/cuda_packed_decoder.py','test2026/cuda/qwen38_5090/run_logged.py','test2026/cuda/qwen38_5090/monitor_native_probe.py']]
raw={p.relative_to(r).as_posix():p.read_bytes() for p in files};build=json.loads((r/'results/layer0-stage-build/manifest.json').read_text());library=hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest();assert library==build['library_sha256']
links={n:hashlib.sha256((r/'results'/n).read_bytes()).hexdigest() for n in ['native-prefix12-internals-evidence.zip','native-controls-abba-evidence.zip','deployed-io-resources-evidence.zip']}
m={'scope':'Original native build and first actual stage diagnostic failure. B1 raw arrays saved; graph replay assertion failed before B2/primary admission. No stage rootcause, fullbatch, serving or timing proof. Original source/runtime retained.','terminal':terminal,'library_sha256':library,'linked_archives':links,'files':{n:hashlib.sha256(b).hexdigest() for n,b in raw.items()}}
out=r/'results/layer0-real-stage-attempt1-evidence.zip';assert not out.exists()
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for n,b in raw.items():z.writestr(n,b)
 z.writestr('evidence-manifest.json',json.dumps(m,indent=2))
print(json.dumps({'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'files':len(raw),'library_sha256':library}))
