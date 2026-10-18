"""Run the frozen full proof; restore normal service in the inner controller."""
from pathlib import Path
import hashlib,json,subprocess,sys,datetime
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-paired-gpu-profile-current';main=b/'work/worldsense-paired-gpu-profile'
frozen=json.loads((w/'learned-identities-before.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(Path(__file__))==frozen['external'][str(Path(__file__))]
assert not(main/'learned-outer-terminal.json').exists(),'No unchanged retry/overwrite'
completed=subprocess.run([sys.executable,str(w/'full_correctness.py')])
(main/'learned-outer-terminal.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=completed.returncode,controller=str(w/'full_correctness.py'),no_performance_claim=True),indent=2)+'\n')
sys.exit(completed.returncode)
