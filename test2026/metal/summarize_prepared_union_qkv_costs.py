"""Frozen QKV component cost ranges; no sum into full-model setup, RSS or startup claims."""
from pathlib import Path
import hashlib,json,sys
base=Path(sys.argv[1]).resolve();result=dict(cohorts=[],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),full_goal_met=False)
for label in ('initial','repeat'):
 folder=base/label;audit=json.loads((folder/'verification.json').read_text());assert audit['complete_learned_profiles']==116
 proof=json.loads((folder/'evidence/manifest.json').read_text());trial=next(iter(proof['jobs']));jobs=folder/'evidence/logs'/trial/'jobs';records=[json.loads(line) for line in (jobs/'numeric.log').read_text().splitlines()];rows=[v for v in records if v['kind']=='numeric'];models=[]
 for name in sorted({v['model'] for v in rows}):
  selected=[v for v in rows if v['model']==name];first=min(v['rows'] for v in selected);packing=[v for v in selected if v['rows']==first and v['prepared']];reused=[v for v in selected if v['rows']!=first and v['prepared']]
  def extent(values):return dict(min=min(values),max=max(values))
  models.append(dict(model=name,admitted_groups=len(packing),first_shape_preparation_ms=extent([v['prepare_ms'] for v in packing]),second_shape_preparation_ms=extent([v['prepare_ms'] for v in reused]),original_weight_upload_ms=extent([v['upload_ms'] for v in packing]),logical_all_admitted_union_weights_bytes=sum(v['extra_union_weight_bytes'] for v in packing),rejected_shape_preparation_ms=[v['prepare_ms'] for v in selected if not v['prepared']]))
 result['cohorts'].append(dict(cohort=label,models=models,numeric_peak_rss_bytes=audit['summary']['numeric']['process_peak_rss_bytes'],timing_peak_rss_bytes=audit['summary']['timing']['process_peak_rss_bytes'],scope='One learned triple at a time, anchor pipeline cache; payloads are estimates, process peaks and clock ranges are component-only.'))
(base/'cost-summary.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
