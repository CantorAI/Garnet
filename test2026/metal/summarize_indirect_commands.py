"""Verify the component gate and recompute separate encoding/wall/GPU means."""
import hashlib,json,statistics,sys
from pathlib import Path
base=Path(sys.argv[1]).resolve();e=base/'evidence'
assert json.loads((e/'exit.json').read_text())['exit_code']==0
statuses=json.loads((e/'statuses.json').read_text());assert len(statuses)==2 and all(s['exit_code']==0 for s in statuses)
monitors=json.loads((e/'probe-monitor.json').read_text());assert monitors and all(m['guard_passed'] for m in monitors)
lines=(e/'probe.log').read_text().splitlines()
rows=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('sample ')]
completed=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('completed ')]
assert len(rows)==1728 and completed==[{'exact_checks':2160}]
assert all(r['exact'] and r['encode_ms']>=0 and r['gpu_ms']>=0 and r['wall_ms']>=r['encode_ms'] for r in rows)
metadata=json.loads((e/'metadata.json').read_text())
source=Path(__file__).resolve().parent
for name,sha in metadata['source_sha256'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==sha,name
profiles=[]
for n in (896,1024,2048):
 for depth in (16,64,256):
  for round_id in (0,1):
   means={}
   for mode,count in [('direct',24),('single',24),('indirect',48)]:
    selected=[r for r in rows if (r['elements'],r['dispatches'],r['round'],r['mode'])==(n,depth,round_id,mode)]
    assert len(selected)==count
    means[mode]={k:statistics.mean(r[k] for r in selected) for k in ('encode_ms','wall_ms','gpu_ms')}
   profiles.append(dict(elements=n,dispatches=depth,round=round_id,means=means,
      indirect_vs_direct_wall_fraction=means['indirect']['wall_ms']/means['direct']['wall_ms'],
      indirect_vs_single_wall_fraction=means['indirect']['wall_ms']/means['single']['wall_ms']))
report=dict(scope='Synthetic dependent FP32 additions only; no model performance claim',exact_checks=2160,timed_records=1728,
    profiles=profiles,source_sha256=metadata['source_sha256'],files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in e.iterdir() if p.is_file()})
(base/'summary.json').write_bytes((json.dumps(report,indent=2)+'\n').encode('utf-8'))
for p in profiles:print(p['elements'],p['dispatches'],p['round'],round(p['indirect_vs_direct_wall_fraction'],3),round(p['indirect_vs_single_wall_fraction'],3))
