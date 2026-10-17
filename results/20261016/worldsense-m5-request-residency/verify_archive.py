"""Verify archived raw-call coverage and immutable artifacts after finalization."""
from pathlib import Path
import csv,hashlib,json
root=Path('/Users/shawn/CantorAI/Garnet/results/20261016/worldsense-m5-request-residency')
manifest=json.loads((root/'file-manifest.json').read_text())
actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='file-manifest.json'}
assert actual==set(manifest),'Manifest must cover exactly all archived files'
for rel,digest in manifest.items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest,rel
assert (root/'dense-words-off.bin').read_bytes()==(root/'dense-words-on.bin').read_bytes()
expected=[]
for cohort in [root,root/'speed',root/'confirmation']:
 if not cohort.exists():continue
 for path in sorted(cohort.glob('*.json')):
  if not path.name.startswith(('control','candidate')):continue
  data=json.loads(path.read_text())
  if not isinstance(data,dict) or 'items' not in data or 'search_plans' not in data:continue
  for category,items in [('frame',data['items']),('query',data['search_plans'])]:
   for index,item in enumerate(items):
    calls=item['result'].get('model_response',{}) if category=='frame' else {'query_plan':item['result'].get('benchmark_model_response',{})}
    for phase,call in calls.items():
     if isinstance(call,dict) and 'model_id' in call:expected.append((str(path.relative_to(root)),category,str(index),phase,call['model_id']))
with (root/'execution-routes.csv').open() as f:rows=list(csv.DictReader(f))
observed=[tuple(r[k] for k in ['artifact','category','sample_index','phase','model_id']) for r in rows]
assert sorted(observed)==sorted(expected),'Route CSV must cover all raw model calls exactly once'
for row in rows:assert row['artifact_sha256']==hashlib.sha256((root/row['artifact']).read_bytes()).hexdigest()
print(json.dumps({'verified_sha256_files':len(manifest),'raw_model_calls_and_route_rows':len(expected),'native_output_words_identical':True,'scope':'Archive integrity and call coverage; not an independent speed or correctness result.'}))
