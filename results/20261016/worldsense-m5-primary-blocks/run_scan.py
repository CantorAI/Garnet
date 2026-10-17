from pathlib import Path
import json,hashlib,subprocess,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI/out/research/worldsense-primary-blocks');freeze=json.loads((w/'execution-freeze.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(r/'scan')==freeze['executable_sha256']
assert all(sha(Path(p))==h for p,h in freeze['dependency_closure_sha256'].items())
assert sha(w/'matrix-selection.json')==freeze['selection_sha256']
cmd=[str(r/'scan'),str(w/'matrix-selection.json'),str(w/'scan-results.json')]
state={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':cmd,'scope':'Native CPU full-corpus codec exactness and byte accounting; no GPU/model performance claim'}
(w/'execution-status.json').write_text(json.dumps(state,indent=2))
with(w/'scan.log').open('w')as f:proc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
state.update(returncode=proc.returncode,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),executable_sha256=sha(r/'scan'),dependency_identities_still_match=all(sha(Path(p))==h for p,h in freeze['dependency_closure_sha256'].items()))
(w/'execution-status.json').write_text(json.dumps(state,indent=2));assert proc.returncode==0
print('NativeCPU full-corpus scan terminal0; no performance claim',flush=True)
