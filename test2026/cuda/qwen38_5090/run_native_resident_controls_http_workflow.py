from pathlib import Path
import hashlib,json,subprocess,sys,time
from urllib.request import urlopen
r=Path('D:/CantorAI/5090-study-20261023');p=r/'Garnet/test2026/cuda/qwen38_5090';sys.path.insert(0,str(p));from validate_native_mtp_http import validate_worker_profile
from run_logged import run
proof=r/'results/warm-mtp-resident-controls-attempt1-verified.json';v=json.loads(proof.read_text(encoding='utf8'));assert v['status']=='resident_controls_correctness_evidence_verified'
assert v['archive_sha256']==hashlib.sha256((r/'results/warm-mtp-resident-controls-attempt1-evidence.zip').read_bytes()).hexdigest()
assert hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest()=='ba67dc9d4f8b9948615e522e3b6ad0aa10ea6f742505465084449fe57ead0b8b'
out=r/'results/native-resident-controls-http-workflow';out.mkdir(exist_ok=False);diagnostic=r/'results/shared-resident-controls-http-diagnostic';pilot=r/'results/native-resident-controls-profile-pilot'
assert not diagnostic.exists() and not pilot.exists();deadline=time.monotonic()+1800
while True:
 try:
  with urlopen('http://127.0.0.1:18082/health',timeout=5) as response:health=json.load(response)
 except Exception:
  if time.monotonic()>=deadline:raise TimeoutError('Inspect original server handle; no restart on this deadline')
  time.sleep(5);continue
 validate_worker_profile(health,3,4,'direct',4,128,True,False,True)
 assert health.get('native_worker',{}).get('resident_request_controls') is True;break
(out/'ready-health.json').write_text(json.dumps(health,indent=2),encoding='utf8')
code=run([sys.executable,str(p/'probe_shared_mtp_sequences.py'),str(r),str(diagnostic),'--joined-mlp'],r/'results/shared-resident-controls-http-diagnostic-ledger','correctness','resident-controls-twenty-pairs',[proof,r/'build/bin/garnet.dll'])
if code:raise SystemExit(code)
subprocess.run([sys.executable,str(r/'run-native-resident-controls-http-candidate.py'),str(r),str(pilot),'--diagnostic',str(diagnostic),'--joined-mlp'],check=True)
(out/'completed.json').write_text(json.dumps({'status':'completed','diagnostic':str(diagnostic),'pilot':str(pilot),'scope':'Resident-controls transport/model proof,20 same-target MTP pairs plus20 old-qualified identities and2 SSE gates precede measured B1 pilot. No actual CUDA batching or2x qualification.'},indent=2),encoding='utf8')
