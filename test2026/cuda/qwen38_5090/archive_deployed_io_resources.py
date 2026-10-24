from pathlib import Path
import hashlib,json,zipfile
r=Path('D:/CantorAI/5090-study-20261023');folders=['deployed-io-resources-attempt1','deployed-io-resources-controller-ledger','deployed-io-resources-parse-ledger'];terminal={}
for n,expected in [(folders[1],1),(folders[2],0)]:
 ev=[json.loads(l) for l in (r/'results'/n/'commands.jsonl').read_text().splitlines()];assert ev[-1]['event']=='finished' and ev[-1]['exit_code']==expected;terminal[n]=ev[-1]
report=json.loads((r/'results/deployed-io-resources-attempt1/corrected-parse.json').read_text());assert report['status']=='deployed_sm120_static_resources_read'
assert hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest()==report['library_sha256']
files=[p for n in folders for p in (r/'results'/n).rglob('*') if p.is_file()];files += [r/n for n in ['run-deployed-io-resources.py','parse-deployed-io-resources.py','archive-deployed-io-resources.py','cuobjdump132-tool.zip','results/native-controls-abba-verified.json']]
raw={p.relative_to(r).as_posix():p.read_bytes() for p in files};manifest={'scope':'Offline actual deployed sm120 CUDA resource metadata, including original parser format failure and corrected raw replay. No native/GPU timing or VRAM-cause proof; no binary/source/driver change.','terminal':terminal,'linked_archives':{'native-controls-abba-evidence.zip':hashlib.sha256((r/'results/native-controls-abba-evidence.zip').read_bytes()).hexdigest()},'library_sha256':report['library_sha256'],'files':{n:hashlib.sha256(b).hexdigest() for n,b in raw.items()}}
out=r/'results/deployed-io-resources-evidence.zip';assert not out.exists()
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for n,b in raw.items():z.writestr(n,b)
 z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
print(json.dumps({'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'files':len(raw)}))
