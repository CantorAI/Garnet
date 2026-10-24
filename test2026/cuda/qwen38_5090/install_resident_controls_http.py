from pathlib import Path
import hashlib,json,socket,subprocess,zipfile
r=Path('D:/CantorAI/5090-study-20261023');out=r/'results/resident-controls-http-source-install'
assert not out.exists()
events=[json.loads(l) for l in (r/'results/warm-mtp-resident-controls-controller-ledger/commands.jsonl').read_text().splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
with socket.socket() as s:assert s.connect_ex(('127.0.0.1',18082))!=0
library=hashlib.sha256((r/'build/bin/garnet.dll').read_bytes()).hexdigest();assert library=='ba67dc9d4f8b9948615e522e3b6ad0aa10ea6f742505465084449fe57ead0b8b'
package=r/'resident-controls-http-install.zip'
with zipfile.ZipFile(package) as z:
 m=json.loads(z.read('manifest.json'))
 for n,v in m.items():
  dest=(r/n).resolve();assert dest.is_relative_to(r.resolve()) and hashlib.sha256(z.read(n)).hexdigest()==v['sha256']
  if 'previous_sha256' in v:assert hashlib.sha256(dest.read_bytes()).hexdigest()==v['previous_sha256'],n
  else:assert not dest.exists(),n
 proof=json.loads(z.read('results/warm-mtp-resident-controls-attempt1-verified.json'));assert proof['status']=='resident_controls_correctness_evidence_verified' and proof['archive_sha256']==hashlib.sha256((r/'results/warm-mtp-resident-controls-attempt1-evidence.zip').read_bytes()).hexdigest()
 out.mkdir()
 for n,v in m.items():
  dest=r/n
  if 'previous_sha256' in v:
   old=out/'backup'/n;old.parent.mkdir(parents=True,exist_ok=True);old.write_bytes(dest.read_bytes())
  dest.write_bytes(z.read(n));assert hashlib.sha256(dest.read_bytes()).hexdigest()==v['sha256']
 (out/'receipt.json').write_text(json.dumps({'status':'installed','library_sha256':library,'package_sha256':hashlib.sha256(package.read_bytes()).hexdigest(),'files':m},indent=2));print('RESIDENT_HTTP_SOURCE_INSTALLED; native binary unchanged; no worker started')
