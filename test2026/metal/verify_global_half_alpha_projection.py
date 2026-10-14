"""Verify full-output/F64 gates and recompute trained-alpha component timing."""
import hashlib,json,statistics,sys
from pathlib import Path
base=Path(sys.argv[1]);e=base/'evidence';files=json.loads((e/'files.json').read_text())
for name,item in files.items():
 p=base/name;assert p.resolve().is_relative_to(base.resolve());raw=p.read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
name=sys.argv[3] if len(sys.argv)>3 else 'global-half-alpha-projections-20261014';job=e/'logs'/name;assert json.loads((job/'exit.json').read_text())['exit_code']==0
j=job/'jobs';meta=json.loads((j/'metadata.json').read_text());stages=json.loads((j/'statuses.json').read_text());assert len(stages)==4 and all(v['exit_code']==0 for v in stages)
for path,sha in meta['source_sha256'].items():assert hashlib.sha256((j/'source-snapshot'/path).read_bytes()).hexdigest()==sha
omitted=json.loads((e/'omitted-matrices.json').read_text());profiles=[];checks=[];excluded=[]
for model,data in meta['assets'].items():
 manifest=json.loads((j/(model+'-manifest.json')).read_text());assert manifest['cases']==data['cases'];cases={c['tensor']:c for c in data['cases']}
 for tensor,c in cases.items():
  p=omitted[model+':'+tensor];assert p['sha256']==c['sha256'] and p['bytes']==c['columns']*c['inner']*2 and p['checkpoint_sha256']==data['checkpoint_sha256']
 rows=[json.loads(v) for v in (j/(model+'-gpu.log')).read_text().splitlines() if v.startswith('{')]
 omitted_cases=[v for v in rows if v['kind']=='excluded'];excluded.extend(omitted_cases);quality=[v for v in rows if v['kind']=='check']
 assert len(quality)==(len(cases)-len(omitted_cases))*len(manifest['rows']);assert json.loads((j/(model+'-gpu-monitor.json')).read_text())
 for q in quality:
  assert q['learned_bits_exact'] and q['padding_preserved'] and q['input_variants']==4 and q['validated_executions']>=124 and q['common_warm_ms']>=250
  assert all(0<=q[k]<=1 for k in ('comparison_worst_ratio','control_fp64_worst_ratio','candidate_fp64_worst_ratio'))
  timing=[v for v in rows if v['kind']=='timing' and v['tensor']==q['tensor'] and v['rows']==q['rows']];assert len(timing)==8
  for t in timing:assert all(len(t[k])==12 for k in ('encode_ms','wall_ms','gpu_ms'))
  means={policy:{k:statistics.mean(statistics.median(v[k]) for v in timing if v['candidate']==policy) for k in ('encode_ms','wall_ms','gpu_ms')} for policy in (False,True)}
  profiles.append(dict(model=model,tensor=q['tensor'],rows=q['rows'],means={'control':means[False],'candidate':means[True]},ratios={k:means[True][k]/means[False][k] for k in means[False]}));checks.append(q)
assert len(checks)==260 and len(excluded)==2
report=dict(verified_files=len(files),checked_profiles=len(checks),excluded=excluded,validated_executions=sum(q['validated_executions'] for q in checks),timing_windows=len(checks)*8,fp64_worst_ratio=max(q['candidate_fp64_worst_ratio'] for q in checks),comparison_worst_ratio=max(q['comparison_worst_ratio'] for q in checks),profiles=profiles,full_model_gain_established=False)
Path(sys.argv[2]).write_bytes((json.dumps(report,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in report.items() if k!='profiles'}))
