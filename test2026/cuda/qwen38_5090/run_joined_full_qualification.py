from pathlib import Path
import hashlib,json,sys,time
r=Path('D:/CantorAI/5090-study-20261023');ledger=r/'results/joined-decoder-build-ledger/commands.jsonl'
print('WAITING_FOR_ORIGINAL_JOINED_FULL_BUILD',flush=True)
while True:
 events=[json.loads(line) for line in ledger.read_text().splitlines()]
 if events[-1]['event']=='finished':break
 time.sleep(5)
assert events[-1]['exit_code']==0,'Original full build failed'
identity={'source':json.loads((r/'study-state.json').read_text()),'library_sha256':hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest(),'prior_component_library_sha256':hashlib.sha256((r/'build/qualified-before-joined-decoder/garnet.dll').read_bytes()).hexdigest()}
assert identity['prior_component_library_sha256']=='e194bcc2cfaf923676de9dfd44ae99f8bb8fa508e00b4746d8f4639883a00209'
(r/'results/joined-decoder-library-identity.json').write_text(json.dumps(identity,indent=2))
sys.path.insert(0,str(r/'Garnet/test2026/cuda/qwen38_5090'))
from run_logged import run
steps=[('joined-state-original','test-joined-state-original.bat','joined-program-equivalence/original-report.json','passed'),('joined-state-candidate','test-joined-state-candidate.bat','joined-program-equivalence/joined-report.json','passed'),('joined-decoder-full','test-joined-decoder-full.bat','joined-decoder-full-qualification/report.json','smoke_passed')]
for label,batch,report_name,status in steps:
 command=['C:/Python312/python.exe',str(r/'monitor_native_probe.py'),'--study',str(r),'--output',str(r/('results/'+label+'-resources')),'--','cmd.exe','/d','/c',str(r/batch)]
 code=run(command,r/('results/'+label+'-ledger'),'correctness',label,[r/batch,r/'joined-decoder-source.zip'])
 if code:raise SystemExit(code)
 report=json.loads((r/'results'/report_name).read_text());assert report['status']==status
 assert report['library_sha256']==identity['library_sha256']
 if label.startswith('joined-state'):
  nv=192 if label=='joined-state-original' else 128;graph=report['program']['graph']
  assert graph['native_nvfp4_projection_weights']==nv and graph['native_fp8_projection_weights']==208
  assert report['pool']['compressed_projection_pool_entries']==nv+208
print('FULL_JOINED_STATE_AND_GENERATION_GATES_PASSED; no benchmark or broad quality claim',flush=True)
