"""Verify all full vision projection outputs/F64 and recompute component clocks."""
import hashlib,json,statistics,sys
from pathlib import Path
base=Path(sys.argv[1]).resolve();e=base/'evidence';files=json.loads((e/'files.json').read_text())
for name,item in files.items():
 p=base/name;assert p.resolve().is_relative_to(base);raw=p.read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
omitted=json.loads((e/'omitted-matrices.json').read_text());jobs=json.loads((e/'jobs.json').read_text());runs=[]
assert jobs
for name,status in sorted(jobs.items()):
 assert status['exit_code']==0;j=e/'logs'/name/'jobs';meta=json.loads((j/'metadata.json').read_text());stages=json.loads((j/'statuses.json').read_text())
 assert len(stages)==2 and all(v['exit_code']==0 for v in stages)
 for stage in stages:assert json.loads((j/(stage['name']+'-monitor.json')).read_text())
 for path,sha in meta['source_sha256'].items():assert hashlib.sha256((j/'source-snapshot'/path).read_bytes()).hexdigest()==sha
 data=meta['assets']['Qwen3.5-0.8B'];manifest=json.loads((j/'manifest.json').read_text());assert manifest['cases']==data['cases']
 cases={c['tensor']:c for c in data['cases']};assert len(cases)==len(data['cases'])==50
 for tensor,c in cases.items():
  p=omitted['Qwen3.5-0.8B:'+tensor];assert p['sha256']==c['sha256'] and p['bytes']==c['columns']*c['inner']*2 and p['checkpoint_sha256']==data['checkpoint_sha256']
  assert c['rows']==([54,108] if tensor.startswith('model.visual.merger.') else [216,432])
 records=[json.loads(v) for v in (j/'vision-gpu.log').read_text().splitlines() if v.startswith('{')];assert records[-1]['kind']=='completed'
 excluded=[r for r in records if r['kind']=='excluded'];quality=[r for r in records if r['kind']=='check'];profiles=[]
 rejected={v['tensor'] for v in excluded};assert len(rejected)==len(excluded) and rejected<=set(cases)
 assert {(q['tensor'],q['rows']) for q in quality}=={(c['tensor'],r) for c in cases.values() if c['tensor'] not in rejected for r in c['rows']}
 assert len(quality)==2*(len(cases)-len(excluded))
 for q in quality:
  c=cases[q['tensor']];assert q['columns']==c['columns'] and q['inner']==c['inner']
  assert q['learned_bits_exact'] and q['padding_preserved'] and q['input_variants']==4 and q['validated_executions']>=124 and q['common_warm_ms']>=250
  assert all(0<=q[k]<=1 for k in ('comparison_worst_ratio','control_fp64_worst_ratio','candidate_fp64_worst_ratio'))
  windows=[v for v in records if v['kind']=='timing' and v['tensor']==q['tensor'] and v['rows']==q['rows']];assert len(windows)==8
  for round_index in (0,1):assert [v['candidate'] for v in windows if v['round']==round_index]==[False,True,True,False]
  assert all(len(v[k])==12 and all(n>=0 for n in v[k]) for v in windows for k in ('encode_ms','wall_ms','gpu_ms'))
  means={policy:{k:statistics.mean(n for v in windows if v['candidate']==policy for n in v[k]) for k in ('encode_ms','wall_ms','gpu_ms')} for policy in (False,True)}
  profiles.append(dict(tensor=q['tensor'],rows=q['rows'],means={'control':means[False],'candidate':means[True]},fractions={k:means[True][k]/means[False][k] for k in means[False]}))
 runs.append(dict(job=name,total_matrices=len(cases),excluded=excluded,profiles=profiles,validated_executions=sum(q['validated_executions'] for q in quality),timed_executions=96*len(quality),fp64_worst_ratio=max((q['candidate_fp64_worst_ratio'] for q in quality),default=None),comparison_worst_ratio=max((q['comparison_worst_ratio'] for q in quality),default=None)))
result=dict(verified_files=len(files),canonical_matrices=len(omitted),scope='Actual FP32 vision MPS versus exact common half alpha; component only',runs=runs,full_model_gain_established=False,full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(dict(verified_files=len(files),canonical_matrices=len(omitted),runs=[{k:v for k,v in r.items() if k not in ('profiles','excluded')}|{'excluded_count':len(r['excluded'])} for r in runs])))
