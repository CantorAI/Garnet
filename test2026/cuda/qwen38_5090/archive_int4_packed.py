from pathlib import Path
import hashlib,json,zipfile
r=Path('D:/CantorAI/5090-study-20261023');folder=r/'results/int4-packed-qualification';resources=r/'results/int4-packed-qualification-resources';ledger=r/'results/int4-packed-qualification-ledger'
d=json.loads((folder/'report.json').read_text());events=[json.loads(line) for line in (ledger/'commands.jsonl').read_text().splitlines()];resource=json.loads((resources/'resource-monitor.json').read_text())
assert d['status']=='smoke_passed' and len(d['cases'])==12 and events[-1]['event']=='finished' and events[-1]['exit_code']==resource['child_exit_code']==0
files=[p for base in (folder,resources,ledger) for p in base.rglob('*') if p.is_file()]
files += [r/name for name in ['results/int4-full-qualification/report.json','study-state.json','source-manifest.json','preparation-manifest.json','test-int4-packed-full.bat','run-int4-after-layer-diagnostic.py','Garnet/test2026/cuda/qwen38_5090/probe_int4_packed_full.py','Garnet/xModel/qwen38/vl_27b/model.py','Garnet/xModel/qwen35/vl_0_8b/cuda_packed_decoder.py','Garnet/xModel/qwen35/vl_0_8b/cuda_packed_program.py','models/Qwen3.8-27B-INT4/config.json','models/Qwen3.8-27B-INT4/model.safetensors.index.json']]
files += list((r/'results/int4-full-qualification').glob('*.f32'))
for key in {row['key'] for row in d['cases']}:
 for kind in ('request','response'):files.append(r/'results/native-mtp3-http'/(key+'-'+kind+'.json'))
raws={p.relative_to(r).as_posix():p.read_bytes() for p in files};out=r/'results/int4-packed-evidence.zip';assert not out.exists()
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for name,raw in raws.items():z.writestr(name,raw)
 z.writestr('evidence-manifest.json',json.dumps({'scope':d['scope'],'binary_sha256':hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest(),'files':{n:hashlib.sha256(raw).hexdigest() for n,raw in raws.items()}},indent=2))
print(json.dumps({'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'entries':len(raws)+1}))
