from pathlib import Path
import hashlib,json,subprocess,zipfile
r=Path('D:/CantorAI/5090-study-20261023');out=r/'results/native-controls-abba-evidence.zip';assert not out.exists()
ledger=r/'results/native-controls-abba-controller-ledger';events=[json.loads(l) for l in (ledger/'commands.jsonl').read_text().splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==0,'Observe original controller terminal0 before archiving complete experiment'
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
folder=r/'results/native-controls-abba';complete=json.loads((folder/'completed.json').read_text());assert complete['status']=='four_legs_finished_needs_independent_reaccounting'
assert [x['leg'] for x in complete['legs']]==['a1','b1','b2','a2'] and all(x['workflow_exit_code']==0 for x in complete['legs'])
files=[p for d in (folder,ledger) for p in d.rglob('*') if p.is_file()];files += [r/n for n in ['run-native-controls-abba.py','controls-abba-methods.zip','controls-abba-experiment-plan.json','archive-native-controls-abba.py']]
raw={p.relative_to(r).as_posix():p.read_bytes() for p in files};links={}
for row in complete['legs']:
 n='native-controls-abba-'+row['leg']+'-http-evidence.zip';h=hashlib.sha256((r/'results'/n).read_bytes()).hexdigest();assert h==row['archive_sha256'];links[n]=h
manifest={'scope':'Four prespecified OFF/ON/ON/OFF B1 legs. Source/runtime/requests and raw quality/SSE/VRAM must be independently reaccounted in each linked archive; telemetry is before/after only. Prior exploratory pilot excluded. No tuned-reference/full-context/broad-quality/trueCUDA batching or2x admission.','terminal':events[-1],'linked_archives':links,'library_sha256':hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest(),'files':{n:hashlib.sha256(b).hexdigest() for n,b in raw.items()}}
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for n,b in raw.items():z.writestr(n,b)
 z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
print(json.dumps({'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'files':len(raw),'linked_archives':links}))
