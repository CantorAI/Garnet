"""Sequential component seed policies; no competing inference or serving claim."""
import hashlib,json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(exist_ok=False)
s=root/'source/test2026/metal';statuses=[];profiles=[]
for seed in (8192,1024,4096,32768):
    destination=out/('seed-'+str(seed))
    code=subprocess.call([sys.executable,str(s/'run_screened_seed_probe.py'),str(root),str(destination),str(seed)])
    statuses.append(dict(seed_rows=seed,exit_code=code))
    (out/'statuses.json').write_bytes((json.dumps(statuses,indent=2)+'\n').encode())
    assert code==0,'Seed probe failed; remaining policies not run'
    inner=json.loads((destination/'summary.json').read_text());assert len(inner['profiles'])==51
    for p in inner['profiles']:profiles.append(dict(seed_rows=seed,**p))
    (out/'summary.json').write_bytes((json.dumps(dict(profiles=profiles,scope=__doc__,full_goal_met=False),indent=2)+'\n').encode())
print('Four exact-seed policies completed sequentially; component only',flush=True)
