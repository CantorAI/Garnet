"""Audit trained-weight projection gates and recompute all observed timings."""
import hashlib,json,math,statistics,sys
from pathlib import Path
base=Path(sys.argv[1]).resolve();manifest=json.loads((base/'manifest.json').read_text());checks=[];profiles=[]
for entry in manifest['files']:
 p=base/entry['path'];assert p.resolve().is_relative_to(base)
 assert p.stat().st_size==entry['bytes'] and hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==entry['sha256']
root=base/'logs/indirect-bf16-projection-20261014';jobs=root/'jobs'
assert json.loads((root/'exit.json').read_text())['exit_code']==0
stages=json.loads((jobs/'statuses.json').read_text());assert len(stages)==4 and all(s['exit_code']==0 for s in stages)
metadata=json.loads((jobs/'metadata.json').read_text())
assert metadata['binary_sha256']==next(e['sha256'] for e in manifest['omitted'] if e['path'].endswith('/indirect_bf16_gpu'))
for name,sha in metadata['source_sha256'].items():assert hashlib.sha256((jobs/'source-snapshot'/name).read_bytes()).hexdigest()==sha
assert hashlib.sha256((jobs/'native-decode.metal').read_bytes()).hexdigest()==metadata['shader_sha256']
total_windows=0;total_references=0;total_validated=0
for model,assets in metadata['assets'].items():
 rows=[json.loads(line) for line in (jobs/(model+'-gpu.log')).read_text().splitlines() if line.startswith('{')]
 lookup={case['name']:case for case in assets['cases']};quality=[r for r in rows if r['kind']=='check'];timings=[r for r in rows if r['kind']=='timing']
 assert len(quality)==len(lookup)*2 and len(timings)==len(quality)*16
 terminal=[r for r in rows if r['kind']=='completed'];assert terminal==[dict(kind='completed',profiles=len(quality),executions=len(quality)*244)]
 monitor=json.loads((jobs/(model+'-gpu-monitor.json')).read_text());assert monitor and all(x['guard_passed'] for x in monitor)
 for q in quality:
  assert q['bits_exact'] and q['padding_preserved'] and q['input_variants']==4 and q['validated_executions']==240
  assert math.isfinite(q['fp64_worst_ratio']) and q['fp64_worst_ratio']<=1
  case=lookup[q['matrix']];checks.append(dict(model=model,**case,**q))
  for pair in ('native','direct_indirect_flag'):
   for round_id in (0,1):
    means={}
    for policy in ('control','candidate'):
     windows=[v for v in timings if (v['matrix'],v['rows'],v['pair'],v['round'],v['policy'])==(q['matrix'],q['rows'],pair,round_id,policy)]
     assert len(windows)==2 and all(w['bits_exact'] for w in windows)
     samples={k:sum([w[k] for w in windows],[]) for k in ('encode_ms','wall_ms','gpu_ms')}
     assert all(len(values)==24 and all(math.isfinite(x) and x>=0 for x in values) for values in samples.values())
     assert all(wall>=encode for wall,encode in zip(samples['wall_ms'],samples['encode_ms']))
     means[policy]={k:statistics.mean(values) for k,values in samples.items()}
    profiles.append(dict(model=model,**case,rows=q['rows'],pair=pair,round=round_id,means=means,
        wall_fraction=means['candidate']['wall_ms']/means['control']['wall_ms'],
        gpu_fraction=means['candidate']['gpu_ms']/means['control']['gpu_ms'],
        encode_fraction=means['candidate']['encode_ms']/means['control']['encode_ms']))
 total_windows+=len(timings);total_references+=len(quality)*4;total_validated+=len(quality)*240
assert len(checks)==68 and total_windows==1088 and total_validated==16320
report=dict(scope='Real BF16 projection components; no full-model speed claim',verified_files=len(manifest['files']),matrices=34,
    reference_profiles=len(checks),fp64_reference_executions=total_references,bit_exact_validated_executions=total_validated,
    timing_windows=total_windows,timed_executions=total_windows*12,fp64_worst_ratio=max(q['fp64_worst_ratio'] for q in checks),
    metadata=metadata,checks=checks,profiles=profiles)
Path(sys.argv[2]).write_bytes((json.dumps(report,indent=2)+'\n').encode())
for pair in ('native','direct_indirect_flag'):
 selected=[p for p in profiles if p['pair']==pair]
 print(json.dumps(dict(pair=pair,profiles=len(selected),pooled_ratios={k:statistics.mean(p['means']['candidate'][k] for p in selected)/statistics.mean(p['means']['control'][k] for p in selected) for k in ('encode_ms','wall_ms','gpu_ms')},
    median_wall_fraction=statistics.median(p['wall_fraction'] for p in selected))))
print(json.dumps({k:v for k,v in report.items() if k not in ('metadata','checks','profiles')}))
