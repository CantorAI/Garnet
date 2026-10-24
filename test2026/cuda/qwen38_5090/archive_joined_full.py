from pathlib import Path
import hashlib,json,zipfile
r=Path('D:/CantorAI/5090-study-20261023');files=[];phases={};failed=False
for label,report_name in [('joined-state-original','joined-program-equivalence/original-report.json'),('joined-state-candidate','joined-program-equivalence/joined-report.json'),('joined-decoder-full','joined-decoder-full-qualification/report.json')]:
 ledger=r/('results/'+label+'-ledger');resources=r/('results/'+label+'-resources')
 if not ledger.exists():assert failed,'Missing phase before a recorded failure';continue
 events=[json.loads(line) for line in (ledger/'commands.jsonl').read_text().splitlines()];assert events[-1]['event']=='finished','Original GPU phase remains live'
 report=json.loads((r/'results'/report_name).read_text());resource=json.loads((resources/'resource-monitor.json').read_text())
 assert events[-1]['exit_code']==resource['child_exit_code'];failed=failed or events[-1]['exit_code']!=0
 phases[label]={'status':report['status'],'exit_code':events[-1]['exit_code']}
 files += [p for base in (ledger,resources) for p in base.rglob('*') if p.is_file()]
for name in ('joined-program-equivalence','joined-decoder-full-qualification','joined-decoder-build-ledger'):
 files += [p for p in (r/'results'/name).rglob('*') if p.is_file()]
files += [r/name for name in ['build/bin/garnet.dll','joined-decoder-source.zip','study-state.json','source-manifest.json','preparation-manifest.json','results/joined-decoder-library-identity.json','results/joined-decoder-source-install.json','build-joined-mlp.bat','test-joined-state-original.bat','test-joined-state-candidate.bat','test-joined-decoder-full.bat','run-joined-full-qualification.py','results/shared-decoder-full-qualification/report.json']]
files += list((r/'results/shared-decoder-full-qualification').glob('*normalized.f32'))
for key in ('s3-text-c1-r0-greedy','s3-image-c1-r0-greedy','red','green','blue','bottom-blue'):
 for kind in ('request','response'):files.append(r/'results/native-mtp3-http'/(key+'-'+kind+'.json'))
raws={p.relative_to(r).as_posix():p.read_bytes() for p in files};out=r/'results/joined-full-evidence.zip';assert not out.exists()
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for name,raw in raws.items():z.writestr(name,raw)
 z.writestr('evidence-manifest.json',json.dumps({'scope':'Original/joined full T128 packed output hash comparison and pinned T128/T4 token/normalized-vector gates. First state chunk retained; remaining full-state chunks retain producer hashes rather than duplicate raw arrays. Not nonzero-state T4 equivalence, MTP, broad quality, long context or performance qualification.','phases':phases,'files':{n:hashlib.sha256(raw).hexdigest() for n,raw in raws.items()}},indent=2))
print(json.dumps({'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'entries':len(raws)+1,'phases':phases}))
