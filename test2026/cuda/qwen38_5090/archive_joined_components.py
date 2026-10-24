from pathlib import Path
import hashlib,json,zipfile
r=Path('D:/CantorAI/5090-study-20261023');files=[];reports={}
for label in ('joined-pool','joined-mlp'):
 folder=r/('results/'+label+'-qualification');ledger=r/('results/'+label+'-qualification-ledger');resources=r/('results/'+label+'-qualification-resources')
 if not ledger.exists():
  assert label=='joined-mlp' and reports['joined-pool']['status']!='passed';continue
 events=[json.loads(line) for line in (ledger/'commands.jsonl').read_text().splitlines()];assert events[-1]['event']=='finished','Original component process remains live'
 d=json.loads((folder/'report.json').read_text());resource=json.loads((resources/'resource-monitor.json').read_text());assert events[-1]['exit_code']==resource['child_exit_code']
 if d['status']=='passed':assert events[-1]['exit_code']==0
 reports[label]={'status':d['status'],'exit_code':events[-1]['exit_code']}
 files += [p for base in (folder,ledger,resources) for p in base.rglob('*') if p.is_file()]
files += [p for p in (r/'results/joined-mlp-build-ledger').rglob('*') if p.is_file()]
files += [r/name for name in ['build/bin/garnet.dll','joined-mlp-source.zip','study-state.json','source-manifest.json','preparation-manifest.json','results/joined-mlp-library-identity.json','results/joined-mlp-source-install.json','build-joined-mlp.bat','test-joined-pool.bat','test-joined-mlp.bat','run-joined-qualification.py']]
raws={p.relative_to(r).as_posix():p.read_bytes() for p in files};out=r/'results/joined-components-evidence.zip';assert not out.exists()
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for name,raw in raws.items():z.writestr(name,raw)
 z.writestr('evidence-manifest.json',json.dumps({'scope':'Joined NVFP4 pool byte/lifetime and actual MLP T4/T128 components only; no full decoder or benchmark. GPU pool raw-byte equality is asserted by retained producer; hashes retained, not duplicate packed checkpoint arrays. MLP F32 output vectors are retained.','reports':reports,'files':{n:hashlib.sha256(raw).hexdigest() for n,raw in raws.items()}},indent=2))
print(json.dumps({'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'entries':len(raws)+1,'reports':reports}))
