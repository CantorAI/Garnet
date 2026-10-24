from pathlib import Path
import hashlib,json,zipfile
r=Path('D:/CantorAI/5090-study-20261023');folder=r/'results/native-trt-layer-profiler-fixed-preflight';report=json.loads((folder/'report.json').read_text())
assert report['status']=='passed'
build=r/'results/profiler-lifecycle-build-ledger';events=[json.loads(line) for line in (build/'commands.jsonl').read_text().splitlines()]
assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
assert report['library_sha256']==hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest()
files=[p for base in (folder,r/'results/native-trt-layer-profiler-fixed-preflight-ledger',r/'results/native-trt-layer-profiler-fixed-preflight-resources',build) for p in base.rglob('*') if p.is_file()]
files += [r/name for name in ['build/bin/garnet.dll','build-profiler-lifecycle.bat','test-trt-layer-profiler-fixed.bat','profiler-lifecycle-source.zip','study-state.json','source-manifest.json','preparation-manifest.json','results/profiler-lifecycle-source-install.json','Garnet/src/backends/tensorrt/trt_builder.cpp','Garnet/src/backends/tensorrt/trt_context_pool.h','Garnet/test2026/cuda/qwen38_5090/probe_trt_layer_profiler.py']]
raws={p.relative_to(r).as_posix():p.read_bytes() for p in files};out=r/'results/profiler-lifecycle-qualified-evidence.zip';assert not out.exists()
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for name,raw in raws.items():z.writestr(name,raw)
 z.writestr('evidence-manifest.json',json.dumps({'scope':'Bounded normalization profiling lifecycle qualification, not full model or performance qualification. Includes failed and successful build launcher ledger.','files':{n:hashlib.sha256(raw).hexdigest() for n,raw in raws.items()}},indent=2))
print(json.dumps({'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'entries':len(raws)+1}))
