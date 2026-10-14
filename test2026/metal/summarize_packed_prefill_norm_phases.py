"""Recompute native prefill phases from complete warm request records."""
import json,statistics,sys
from pathlib import Path
base=Path(sys.argv[1]);outer=json.loads((base/'summary.json').read_text());rows=[]
for p in outer['profiles']:
 records=json.loads((base/Path(p['evidence']).name/'summary.json').read_text())['records']
 for policy in ('control','candidate'):
  warm=[r for r in records if r['request']>0 and r['build_policy']==policy];assert len(warm)==4
  rows.append(dict(model=p['model'],batch=p['batch'],policy=policy,**{key:statistics.mean(r['prefill_profile'][key] for r in warm) for key in ('prepare_ms','encode_ms','gpu_ms','wait_ms')}))
Path(sys.argv[2]).write_bytes((json.dumps(dict(scope=__doc__,profiles=rows),indent=2)+'\n').encode());print('Recomputed',len(rows),'prefill phase profiles')
