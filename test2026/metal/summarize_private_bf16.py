"""Verify storage mode, learned-byte readback, full warmed gates and timings."""
import hashlib,json,math,statistics,sys
from pathlib import Path
base=Path(sys.argv[1]).resolve();proof=json.loads((base/'manifest.json').read_text());profiles=[];checks=[]
for e in proof['files']:
 p=base/e['path'];assert p.resolve().is_relative_to(base)
 assert p.stat().st_size==e['bytes'] and hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==e['sha256']
assert proof['jobs']['private-bf16-probe-20261014']['exit_code']==1
assert proof['jobs']['private-bf16-retry-20261014']['exit_code']==0
failed=base/'logs/private-bf16-probe-20261014/jobs'
assert 'Private upload buffers unavailable' in (failed/'Qwen3-0.6B-gpu.log').read_text()
name=sys.argv[3] if len(sys.argv)>3 else 'private-bf16-retry-20261014'
assert name in ('private-bf16-retry-20261014','private-bf16-confirm-verified-cpu-20261014')
assert proof['jobs'][name]['exit_code']==0
jobs=base/'logs'/name/'jobs';metadata=json.loads((jobs/'metadata.json').read_text())
statuses=json.loads((jobs/'statuses.json').read_text());assert len(statuses)==5 and all(s['exit_code']==0 for s in statuses)
assert metadata['binary_sha256']==next(e['sha256'] for e in proof['omitted'] if e['path'].endswith(name+'/jobs/private_bf16_gpu'))
for name,sha in metadata['source_sha256'].items():assert hashlib.sha256((jobs/'source-snapshot'/name).read_bytes()).hexdigest()==sha
assert hashlib.sha256((jobs/'native-decode.metal').read_bytes()).hexdigest()==metadata['shader_sha256']
total_timed=0;total_validated=0
for model,assets in metadata['assets'].items():
 rows=[json.loads(line) for line in (jobs/(model+'-gpu.log')).read_text().splitlines() if line.startswith('{')]
 quality=[r for r in rows if r['kind']=='check'];timings=[r for r in rows if r['kind']=='timing'];setup=[r for r in rows if r['kind']=='setup'];warm=[r for r in rows if r['kind']=='warmup']
 assert len(quality)==len(assets['cases'])*2 and len(timings)==len(quality)*8 and len(setup)==len(warm)==len(quality)
 assert all(s['actual_storage_mode']==2 and s['private_bytes_exact'] and s['copy_and_readback_ms']>=0 for s in setup)
 assert all(w['elapsed_ms']>=250 and w['validated_executions']>0 for w in warm)
 terminal=next(r for r in rows if r['kind']=='completed');assert terminal['profiles']==len(quality)
 assert terminal['executions']==sum(q['validated_executions']+4 for q in quality)
 monitor=json.loads((jobs/(model+'-gpu-monitor.json')).read_text());assert monitor and all(m['guard_passed'] for m in monitor)
 lookup={c['name']:c for c in assets['cases']}
 for q in quality:
  assert q['bits_exact'] and q['padding_preserved'] and q['input_variants']==4 and 0<=q['fp64_worst_ratio']<=1
  w=next(w for w in warm if (w['matrix'],w['rows'])==(q['matrix'],q['rows']));assert q['validated_executions']==120+w['validated_executions']
  total_validated+=q['validated_executions'];checks.append(dict(model=model,**lookup[q['matrix']],**q))
  for round_id in (0,1):
   means={}
   for policy in ('control','candidate'):
    windows=[r for r in timings if (r['matrix'],r['rows'],r['round'],r['policy'])==(q['matrix'],q['rows'],round_id,policy)]
    assert len(windows)==2 and all(r['bits_exact'] for r in windows)
    data={key:sum([w[key] for w in windows],[]) for key in ('encode_ms','wall_ms','gpu_ms')}
    assert all(len(v)==24 and all(math.isfinite(x) and x>=0 for x in v) for v in data.values())
    assert all(w>=e for w,e in zip(data['wall_ms'],data['encode_ms']));means[policy]={key:statistics.mean(v) for key,v in data.items()};total_timed+=24
   profiles.append(dict(model=model,**lookup[q['matrix']],rows=q['rows'],round=round_id,means=means,wall_fraction=means['candidate']['wall_ms']/means['control']['wall_ms'],gpu_fraction=means['candidate']['gpu_ms']/means['control']['gpu_ms']))
assert len(checks)==68 and total_timed==6528
report=dict(scope='Private/shared BF16 projection components; failed attempt timings discarded',verified_files=len(proof['files']),
    reference_profiles=68,timed_executions=total_timed,bit_exact_validated_executions=total_validated,
    fp64_worst_ratio=max(q['fp64_worst_ratio'] for q in checks),checks=checks,profiles=profiles,metadata=metadata)
Path(sys.argv[2]).write_bytes((json.dumps(report,indent=2)+'\n').encode())
for model in metadata['assets']:
 for batch in (1,2):
  v=[p for p in profiles if p['model']==model and p['rows']==batch]
  print(json.dumps(dict(model=model,batch=batch,pooled_ratios={k:statistics.mean(p['means']['candidate'][k] for p in v)/statistics.mean(p['means']['control'][k] for p in v) for k in ('encode_ms','wall_ms','gpu_ms')},median_wall_fraction=statistics.median(p['wall_fraction'] for p in v))))
print(json.dumps({k:v for k,v in report.items() if k not in ('metadata','checks','profiles')}))
