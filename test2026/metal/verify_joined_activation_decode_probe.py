"""Independent audit of lossless components, exclusions and unchanged numerical gates."""
import hashlib,json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();proof=json.loads((root/'evidence/manifest.json').read_text())
for name,info in proof['files'].items():
 p=root/name;assert p.resolve().is_relative_to(root) and p.stat().st_size==info['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256']
assert proof['jobs']['joined-activation-decode-probe-20261014']['exit_code']==1
assert proof['jobs']['joined-activation-decode-repair-probe-20261014']['exit_code']==0
failed=root/'evidence/logs/joined-activation-decode-probe-20261014/jobs';jobs=root/'evidence/logs/joined-activation-decode-repair-probe-20261014/jobs'
assert 'Wide-exponent input was admitted' in (failed/'Qwen3-0.6B-admission.log').read_text()
old=(failed/'source-snapshot/test2026/metal/joined_activation_decode_gpu.mm').read_bytes();new=(jobs/'source-snapshot/test2026/metal/joined_activation_decode_gpu.mm').read_bytes()
assert old.replace(b'if(rejection&&j%101==0)x=std::ldexp(x,-30);',b'if(rejection&&j==1)x=std::ldexp(1.0f,-60);')==new
meta=json.loads((jobs/'metadata.json').read_text());summary=json.loads((jobs/'summary.json').read_text())
for name,sha in meta['source_sha256'].items():assert hashlib.sha256((jobs/'source-snapshot'/name).read_bytes()).hexdigest()==sha
assert hashlib.sha256((jobs/'native-decode.metal').read_bytes()).hexdigest()==meta['native_shader_sha256']
binary=proof['omitted_binaries']['logs/joined-activation-decode-repair-probe-20261014/jobs/joined_activation_decode_gpu'];assert binary['sha256']==meta['binary_sha256']
stages=json.loads((jobs/'statuses.json').read_text());assert len(stages)==7 and all(v['exit_code']==0 for v in stages)
for stage in stages:assert json.loads((jobs/(stage['name']+'-monitor.json')).read_text())
rows=[];exclusions=[];validated=0;timed=0
for model,assets in meta['assets'].items():
 gate=[json.loads(v) for v in (jobs/(model+'-admission.log')).read_text().splitlines() if v.startswith('{')];assert any(v['kind']=='admission' and v['unsafe_input_rejected'] for v in gate)
 data=[json.loads(v) for v in (jobs/(model+'-gpu.log')).read_text().splitlines() if v.startswith('{')];excluded=[v for v in data if v['kind']=='excluded'];checks=[v for v in data if v['kind']=='check']
 assert len(checks)==2*(len(assets['cases'])-len(excluded))
 assert {v['matrix'] for v in checks}|{v['matrix'] for v in excluded}=={c['name'] for c in assets['cases']}
 exclusions.extend(dict(model=model,**v) for v in excluded)
 for check in checks:
  name=check['matrix'];batch=check['rows'];assert check['weight_bits_exact'] and check['activation_bits_exact'] and check['output_padding_preserved'] and check['input_variants']==4 and check['validated_executions']>=130
  assert all(check[k]<=1 for k in ('comparison_worst_ratio','control_fp64_worst_ratio','candidate_fp64_worst_ratio'));validated+=check['validated_executions']
  warm=next(v for v in data if v['kind']=='warmup' and v['matrix']==name and v['rows']==batch);assert warm['elapsed_ms']>=250
  windows=[v for v in data if v['kind']=='timing' and v['matrix']==name and v['rows']==batch];assert len(windows)==8
  for round in (0,1):
   group=sorted((v for v in windows if v['round']==round),key=lambda v:v['window']);assert [v['window'] for v in group]==list(range(4)) and [v['candidate'] for v in group]==[False,True,True,False]
  assert all(len(v[k])==12 and all(x>=0 for x in v[k]) for v in windows for k in ('encode_ms','wall_ms','gpu_ms'));timed+=96
  means={policy:{metric:statistics.mean(x for v in windows if v['candidate']==candidate for x in v[metric]) for metric in ('encode_ms','wall_ms','gpu_ms')} for policy,candidate in (('control',False),('candidate',True))}
  reported=next(v for v in summary['profiles'] if v['model']==model and v['matrix']==name and v['batch']==batch)
  for policy in means:
   for metric in means[policy]:assert abs(means[policy][metric]-reported['means'][policy][metric])<1e-12
  assert abs(reported['gpu_fraction']-means['candidate']['gpu_ms']/means['control']['gpu_ms'])<1e-12 and abs(reported['wall_fraction']-means['candidate']['wall_ms']/means['control']['wall_ms'])<1e-12
  case=next(c for c in assets['cases'] if c['name']==name);rows.append(dict(**reported,tensor=case['tensor'],inner=case['inner'],columns=case['columns']))
assert len(rows)==64 and timed==6144 and exclusions==summary['explicit_weight_exclusions']
pools=[]
for model in meta['assets']:
 for batch in (1,2):
  group=[r for r in rows if r['model']==model and r['batch']==batch and not r['tensor'].startswith('synthetic') and r['columns']>256]
  pools.append(dict(model=model,batch=batch,matrix_count=len(group),gpu_fraction=sum(r['means']['candidate']['gpu_ms'] for r in group)/sum(r['means']['control']['gpu_ms'] for r in group),wall_fraction=sum(r['means']['candidate']['wall_ms'] for r in group)/sum(r['means']['control']['wall_ms'] for r in group)))
result=dict(verified_files=len(proof['files']),matrix_batch_checks=len(rows),validated_executions=validated,timed_executions=timed,weight_exclusions=exclusions,unweighted_trained_large_matrix_pools=pools,
 candidate_fp64_worst_ratio=max(r['checks']['candidate_fp64_worst_ratio'] for r in rows),comparison_worst_ratio=max(r['checks']['comparison_worst_ratio'] for r in rows),scope='Component only; exact representations and bounded products; no full inference or vLLM gain',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
