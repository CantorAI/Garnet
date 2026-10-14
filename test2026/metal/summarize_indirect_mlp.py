"""Verify original-kernel MLP replay and recompute separate timing clocks."""
import hashlib,json,math,statistics,sys
from pathlib import Path
base=Path(sys.argv[1]).resolve();proof=json.loads((base/'manifest.json').read_text())
for e in proof['files']:
 p=base/e['path'];assert p.resolve().is_relative_to(base)
 assert p.stat().st_size==e['bytes'] and hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==e['sha256']
name=sys.argv[3] if len(sys.argv)>3 else 'indirect-mlp-probe-20261014'
assert name in ('indirect-mlp-probe-20261014','indirect-mlp-confirm-20261014')
root=base/'logs'/name;jobs=root/'jobs'
assert json.loads((root/'exit.json').read_text())['exit_code']==0
stages=json.loads((jobs/'statuses.json').read_text());assert len(stages)==2 and all(x['exit_code']==0 for x in stages)
metadata=json.loads((jobs/'metadata.json').read_text());assert len(metadata['assets'])==3
for name,sha in metadata['source_sha256'].items():assert hashlib.sha256((jobs/'source-snapshot'/name).read_bytes()).hexdigest()==sha
assert hashlib.sha256((jobs/'native-mlp.metal').read_bytes()).hexdigest()==metadata['shader_sha256']
assert hashlib.sha256((jobs/'mlp-assets.json').read_bytes()).hexdigest()==metadata['mlp_assets_sha256']
assert metadata['binary_sha256']==next(e['sha256'] for e in proof['omitted'] if e['path'].endswith('/indirect_mlp_gpu'))
monitor=json.loads((jobs/'gpu-monitor.json').read_text());assert monitor and all(x['guard_passed'] for x in monitor)
rows=[json.loads(line) for line in (jobs/'gpu.log').read_text().splitlines() if line.startswith('{')]
assert [r for r in rows if r['kind']=='completed']==[dict(kind='completed',profiles=6,executions=2184)]
checks=[r for r in rows if r['kind']=='check'];timings=[r for r in rows if r['kind']=='timing'];assert len(checks)==6 and len(timings)==144
profiles=[]
for model in metadata['assets']:
 for batch in (1,2):
  q=next(r for r in checks if (r['model'],r['rows'])==(model,batch))
  assert q['bits_exact'] and q['padding_preserved'] and q['input_variants']==4 and q['validated_executions']==360
  assert 0<=q['projection_fp64_worst_ratio']<=1 and 0<=q['silu_worst_ratio']<=1
  for pair in (0,1,2):
   for round_id in (0,1):
    means={}
    for policy in ('control','candidate'):
     windows=[r for r in timings if (r['model'],r['rows'],r['pair'],r['round'],r['policy'])==(model,batch,pair,round_id,policy)]
     assert len(windows)==2 and all(r['bits_exact'] for r in windows)
     values={key:sum([w[key] for w in windows],[]) for key in ('encode_ms','wall_ms','gpu_ms')}
     assert all(len(v)==24 and all(math.isfinite(x) and x>=0 for x in v) for v in values.values())
     assert all(w>=e for w,e in zip(values['wall_ms'],values['encode_ms']))
     means[policy]={key:statistics.mean(v) for key,v in values.items()}
    profiles.append(dict(model=model,batch=batch,pair=pair,round=round_id,means=means,
       wall_fraction=means['candidate']['wall_ms']/means['control']['wall_ms'],
       gpu_fraction=means['candidate']['gpu_ms']/means['control']['gpu_ms'],
       encode_fraction=means['candidate']['encode_ms']/means['control']['encode_ms']))
report=dict(scope='First-layer four-kernel MLP component; no full-model speed claim',verified_files=len(proof['files']),
    reference_profiles=6,baseline_executions=24,bit_exact_validated_executions=2160,timed_executions=1728,
    checks=checks,profiles=profiles,metadata=metadata)
Path(sys.argv[2]).write_bytes((json.dumps(report,indent=2)+'\n').encode())
for p in profiles:print(p['model'],p['batch'],p['pair'],p['round'],round(p['wall_fraction'],4),round(p['encode_fraction'],4),round(p['gpu_fraction'],4))
