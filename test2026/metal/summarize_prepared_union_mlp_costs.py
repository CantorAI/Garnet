"""Summarize frozen component-only clocks and payloads; no startup or model speed inference."""
from pathlib import Path
import hashlib,json,sys
base=Path(sys.argv[1]).resolve();result=dict(cohorts=[],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),full_goal_met=False)
for label in ('initial','repeat'):
 folder=base/label;audit=json.loads((folder/'verification.json').read_text());assert audit['all_layer_profiles']==152
 proof=json.loads((folder/'evidence/manifest.json').read_text());trial=next(iter(proof['jobs']));jobs=folder/'evidence/logs'/trial/'jobs'
 records=[json.loads(line) for line in (jobs/'numeric.log').read_text().splitlines()];rows=[v for v in records if v['kind']=='numeric'];models=[]
 for name in sorted({v['model'] for v in rows}):
  selected=[v for v in rows if v['model']==name];first=min(v['rows'] for v in selected);packing=[v for v in selected if v['rows']==first and v['prepared']];reused=[v for v in selected if v['rows']!=first and v['prepared']]
  def extent(values):return dict(min=min(values),max=max(values))
  models.append(dict(model=name,admitted_layers=len(packing),first_shape_preparation_ms=extent([v['prepare_ms'] for v in packing]),second_shape_preparation_ms=extent([v['prepare_ms'] for v in reused]),pair_upload_ms=extent([v['upload_ms'] for v in packing]),logical_all_admitted_union_weights_bytes=sum(v['extra_union_weight_bytes'] for v in packing),rejected_shape_preparation_ms=[v['prepare_ms'] for v in selected if not v['prepared']]))
 result['cohorts'].append(dict(cohort=label,models=models,numeric_peak_rss_bytes=audit['summary']['numeric']['process_peak_rss_bytes'],timing_peak_rss_bytes=audit['summary']['timing']['process_peak_rss_bytes'],scope='Fresh per-pair contexts with anchor pipeline cache; ranges and summed logical payloads are not full-model initialization/RSS.'))
(base/'cost-summary.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
