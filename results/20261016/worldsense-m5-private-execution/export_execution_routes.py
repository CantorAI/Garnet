"""Export every recorded cohort call's cumulative route counters; never sum as per-call counts."""
from pathlib import Path
import sys,csv,json,hashlib
root=Path(sys.argv[1]);output=Path(sys.argv[2]);rows=[]
fields=['artifact','artifact_sha256','worker','category','sample_index','phase','warmup','repetition','model_id','status','private_execution_runs','scoped_execution_runs','head_private_storage','vision_materialized_weight_slots','vision_pooled_weight_slots','vision_raw_weight_payload_bytes','vision_weight_expansion_dispatches','metal_allocated_bytes','task_physical_footprint_bytes']
for cohort in [root,root/'speed',root/'confirmation']:
 if not cohort.exists():continue
 for path in sorted(cohort.glob('*.json')):
  if not path.name.startswith(('control','candidate')):continue
  raw=path.read_bytes();data=json.loads(raw)
  if not isinstance(data,dict)or 'items'not in data or 'search_plans'not in data:continue
  for category,items in [('frame',data['items']),('query',data['search_plans'])]:
   for index,item in enumerate(items):
    calls=item['result'].get('model_response',{})if category=='frame'else {'query_plan':item['result'].get('benchmark_model_response',{})}
    for phase,call in calls.items():
     if not isinstance(call,dict)or 'model_id'not in call:continue
     timings=call.get('timings',{});memory=timings.get('memory_after_decode',{});row={k:timings.get(k,'')for k in fields}
     row.update(artifact=str(path.relative_to(root)),artifact_sha256=hashlib.sha256(raw).hexdigest(),worker=data.get('label',path.stem),category=category,sample_index=index,phase=phase,warmup=item.get('warmup'),repetition=item.get('repetition'),model_id=call['model_id'],status=call.get('status',''),metal_allocated_bytes=memory.get('metal_allocated_bytes',''),task_physical_footprint_bytes=memory.get('task_physical_footprint_bytes',''));rows.append(row)
with output.open('w',newline='')as f:
 writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
print(json.dumps(dict(rows=len(rows),scope='All first/warm cohort calls. Private/scoped counters cumulative per native object only with PRIVATE_EXECUTION capability; zero in disabled builds is not universal proof of no autorelease scope. Memory postdecode samples not peaks.')))
