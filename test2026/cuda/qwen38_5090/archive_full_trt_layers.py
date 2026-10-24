from pathlib import Path
import hashlib,json,zipfile
r=Path('D:/CantorAI/5090-study-20261023');folder=r/'results/shared-trt-layer-diagnostic';ledger=r/'results/shared-trt-layer-diagnostic-ledger';resources=r/'results/shared-trt-layer-diagnostic-resources'
events=[json.loads(line) for line in (ledger/'commands.jsonl').read_text().splitlines()]
assert events[-1]['event']=='finished','Original full model diagnostic remains live'
d=json.loads((folder/'report.json').read_text());resource=json.loads((resources/'resource-monitor.json').read_text())
assert events[-1]['exit_code']==resource['child_exit_code']
if d['status']=='passed':assert resource['child_exit_code']==0 and len(d['cases'])==12 and d['layers']
files=[p for base in (folder,ledger,resources) for p in base.rglob('*') if p.is_file()]
files += [r/name for name in ['study-state.json','source-manifest.json','preparation-manifest.json','results/profiler-lifecycle-source-install.json','results/native-trt-layer-profiler-fixed-preflight/report.json','test-shared-trt-layers.bat','run-full-profiler-after-qualification.py','Garnet/test2026/cuda/qwen38_5090/probe_shared_trt_layers.py','Garnet/src/backends/tensorrt/trt_builder.cpp','Garnet/src/backends/tensorrt/trt_context_pool.h','Garnet/xModel/qwen35/vl_0_8b/cuda_packed_decoder.py','Garnet/xModel/qwen35/vl_0_8b/cuda_packed_program.py','Garnet/xModel/qwen35/vl_0_8b/cuda_packed_replay.py']]
for key in ['s3-text-c1-r0','s3-image-c1-r0','red','green','blue','bottom-blue']:
 for kind in ('request','response'):files.append(r/'results/shared-mtp-sequence-diagnostic'/(key+'-target-'+kind+'.json'))
raws={p.relative_to(r).as_posix():p.read_bytes() for p in files};out=r/'results/shared-trt-layer-diagnostic-evidence.zip';assert not out.exists()
identity={'library_sha256':hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest(),'qualified_binary_archive_sha256':hashlib.sha256((r/'results/profiler-lifecycle-qualified-evidence.zip').read_bytes()).hexdigest()}
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for name,raw in raws.items():z.writestr(name,raw)
 z.writestr('evidence-manifest.json',json.dumps({'scope':d['scope'],'status':d['status'],'binary':identity,'files':{n:hashlib.sha256(raw).hexdigest() for n,raw in raws.items()}},indent=2))
print(json.dumps({'status':d['status'],'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'entries':len(raws)+1}))
