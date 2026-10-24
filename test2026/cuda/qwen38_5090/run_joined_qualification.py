from pathlib import Path
import hashlib,json,sys,time
r=Path('D:/CantorAI/5090-study-20261023');ledger=r/'results/joined-mlp-build-ledger/commands.jsonl'
print('WAITING_FOR_ORIGINAL_JOINED_BUILD',flush=True)
while True:
 events=[json.loads(line) for line in ledger.read_text().splitlines()]
 if events[-1]['event']=='finished':break
 time.sleep(5)
assert events[-1]['exit_code']==0,'Build failed; no GPU qualification launched'
cache=(r/'build/CMakeCache.txt').read_text();assert 'GARNET_STAGE_VENDOR_DLLS:BOOL=OFF' in cache
identity={'source':json.loads((r/'study-state.json').read_text()),'library_sha256':hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest(),'preserved_qualified_library_sha256':hashlib.sha256((r/'build/qualified-before-joined-mlp/garnet.dll').read_bytes()).hexdigest(),'vendor_staging':'OFF; reuse the previously qualified study dependencies'}
assert identity['preserved_qualified_library_sha256']=='9d82a3dc80e1817a50931ef130e2a6b3023248514687eddfd8f2493ee0dbb0af'
assert identity['library_sha256']!=identity['preserved_qualified_library_sha256']
(r/'results/joined-mlp-library-identity.json').write_text(json.dumps(identity,indent=2))
sys.path.insert(0,str(r/'Garnet/test2026/cuda/qwen38_5090'))
from run_logged import run
for label,batch in [('joined-pool','test-joined-pool.bat'),('joined-mlp','test-joined-mlp.bat')]:
 command=['C:/Python312/python.exe',str(r/'monitor_native_probe.py'),'--study',str(r),'--output',str(r/('results/'+label+'-qualification-resources')),'--','cmd.exe','/d','/c',str(r/batch)]
 code=run(command,r/('results/'+label+'-qualification-ledger'),'correctness',label+'-component-only',[r/batch,r/'joined-mlp-source.zip'])
 if code:raise SystemExit(code)
 assert json.loads((r/('results/'+label+'-qualification/report.json')).read_text())['status']=='passed'
print('JOINED_POOL_AND_MLP_COMPONENTS_QUALIFIED; no full-model or speed claim',flush=True)
