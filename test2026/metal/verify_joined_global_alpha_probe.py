"""Audit trained pair identities, full output/F64 gates and ABBA timing windows."""
import hashlib,json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();e=root/'evidence';files=json.loads((e/'files.json').read_text())
for name,item in files.items():
 p=root/name;assert p.resolve().is_relative_to(root);raw=p.read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
proof=json.loads((e/'omitted-matrices.json').read_text());jobs=json.loads((e/'jobs.json').read_text());results=[]
assert len(proof)==3 and jobs
for job,status in jobs.items():
 assert status['exit_code']==0
 folder=e/'logs'/job/'jobs';meta=json.loads((folder/'metadata.json').read_text())
 for name,sha in meta['source_sha256'].items():assert hashlib.sha256((folder/'source-snapshot'/name).read_bytes()).hexdigest()==sha
 statuses=json.loads((folder/'statuses.json').read_text());assert len(statuses)==4 and all(v['exit_code']==0 for v in statuses)
 for stage in statuses:assert json.loads((folder/(stage['name']+'-monitor.json')).read_text())
 profiles=[];excluded=[];validations=0
 for model,data in meta['assets'].items():
  assert data['checkpoint_sha256']==proof[model]['checkpoint_sha256'] and data['joined']['sha256']==proof[model]['joined_sha256'] and data['members']==proof[model]['members']
  manifest=json.loads((folder/(model+'-manifest.json')).read_text());assert len(manifest['cases'])==1 and manifest['cases'][0]==data['joined']
  rows=manifest['rows'];assert rows==[8,16,22,24,26,32,44,48,52,64,80,128,160]
  records=[json.loads(line) for line in (folder/(model+'-gpu.log')).read_text().splitlines() if line.startswith('{')]
  assert records[-1]['kind']=='completed';checks=[v for v in records if v['kind']=='check'];skipped=[v for v in records if v['kind']=='excluded']
  if skipped:assert len(skipped)==1 and not checks;excluded.extend(skipped);continue
  assert {v['rows'] for v in checks}==set(rows) and len(checks)==13
  for check in checks:
   assert check['learned_bits_exact'] and check['padding_preserved'] and check['input_variants']==4 and check['common_warm_ms']>=250
   assert all(check[k]<=1 for k in ('comparison_worst_ratio','control_fp64_worst_ratio','candidate_fp64_worst_ratio'))
   windows=[v for v in records if v['kind']=='timing' and v['rows']==check['rows']];assert len(windows)==8
   for round_index in (0,1):assert [v['candidate'] for v in windows if v['round']==round_index]==[False,True,True,False]
   assert all(len(v[k])==12 and all(n>=0 for n in v[k]) for v in windows for k in ('encode_ms','wall_ms','gpu_ms'))
   means={str(policy):{key:statistics.mean(n for v in windows if v['candidate']==policy for n in v[key]) for key in ('encode_ms','wall_ms','gpu_ms')} for policy in (False,True)}
   profiles.append(dict(model=model,rows=check['rows'],means=means,fractions={k:means['True'][k]/means['False'][k] for k in means['True']},check=check));validations+=check['validated_executions']
  assert records[-1]['executions']>=sum(v['validated_executions'] for v in checks)
 results.append(dict(job=job,profiles=profiles,excluded=excluded,validated_executions=validations,timed_executions=96*len(profiles)))
result=dict(verified_files=len(files),canonical_pairs=len(proof),runs=results,scope='Two common-alpha products versus one joined product/output views; component only',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps(dict(verified_files=len(files),canonical_pairs=len(proof),runs=[dict(job=v['job'],profiles=len(v['profiles']),excluded=len(v['excluded']),validated_executions=v['validated_executions']) for v in results])))
