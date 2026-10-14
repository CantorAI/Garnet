"""Recompute observed request phases; vision time subtracts separately timed reset."""
import json,statistics,sys
from pathlib import Path
folder=Path(sys.argv[1]);summary=json.loads((folder/'summary.json').read_text());rows=[]
for p in summary['profiles']:
 records=json.loads((folder/Path(p['evidence']).name/'summary.json').read_text())['records']
 if p['model']!='Qwen3.5-0.8B':continue
 for policy in ('control','candidate'):
  warm=[r for r in records if r['build_policy']==policy and r['request']>0];assert len(warm)==4
  row=dict(model=p['model'],batch=p['batch'],policy=policy,ttft_ms=statistics.mean(r['ttft_seconds'] for r in warm)*1000,
           reset_ms=statistics.mean(r['reset_seconds'] for r in warm)*1000,
           vision_encode_elapsed_ms=statistics.mean(r['vision_seconds']-r['reset_seconds'] for r in warm)*1000,
           common_preprocessing_ms=statistics.mean(r['common_preprocessing_seconds'] for r in warm)*1000,
           prefill_wall_ms=statistics.mean(r['prefill_wall_seconds'] for r in warm)*1000)
  rows.append(row)
result=dict(scope='Observed wall phases; vision excludes reset by same-origin clock subtraction; no isolated GPU attribution',profiles=rows)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
