"""Recompute reset and first-use phases from complete native ABBA records."""
import json,statistics,sys
from pathlib import Path
folder=Path(sys.argv[1]);outer=json.loads((folder/'summary.json').read_text());profiles=[]
assert len(outer['profiles'])==6
for p in outer['profiles']:
 records=json.loads((folder/Path(p['evidence']).name/'summary.json').read_text())['records'];assert len(records)==12
 for policy in ('control','candidate'):
  warm=[r for r in records if r['build_policy']==policy and r['request']>0];first=[r for r in records if r['build_policy']==policy and r['request']==0];assert len(warm)==4 and len(first)==2
  value=dict(model=p['model'],batch=p['batch'],policy=policy,warm_requests=4,
   ttft_ms=statistics.mean(r['ttft_seconds'] for r in warm)*1000,
   reset_ms=statistics.mean(r['reset_seconds'] for r in warm)*1000,
   cleared_bytes=statistics.mean(r['dirty_prefix_reset_bytes'] for r in warm),
   bounded_states=statistics.mean(r['dirty_prefix_reset_states'] for r in warm),
   fallback_states=statistics.mean(r['dirty_prefix_reset_fallback_states'] for r in warm),
   decode_tps_per_row=statistics.mean(r['decode_wall_steps_per_second'] for r in warm),
   first_request_after_init_ttft_ms=statistics.mean(r['ttft_seconds'] for r in first)*1000,
   initialization_s=statistics.mean(r['initialization_seconds'] for r in first))
  if p['model']=='Qwen3.5-0.8B':
   value.update(vision_encode_elapsed_ms=statistics.mean(r['vision_seconds']-r['reset_seconds'] for r in warm)*1000,
    preprocessing_ms=statistics.mean(r['common_preprocessing_seconds'] for r in warm)*1000,
    prefill_wall_ms=statistics.mean(r['prefill_wall_seconds'] for r in warm)*1000)
  profiles.append(value)
result=dict(scope='Observed request wall phases; no isolated GPU attribution; init and first request separate; caches not flushed',profiles=profiles)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
