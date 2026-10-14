"""Audit archived fusion gates and recompute alternating-window timings."""
import hashlib,json,statistics,struct,sys
from pathlib import Path
base=Path(sys.argv[1]).resolve();e=base/'evidence'
files=json.loads((e/'files.json').read_text())
for name,item in files.items():
 p=base/name;assert p.resolve().is_relative_to(base);raw=p.read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
job=e/'logs'/sys.argv[3];assert json.loads((job/'exit.json').read_text())['exit_code']==0
j=job/'jobs';meta=json.loads((j/'metadata.json').read_text())
for name,sha in meta['source_sha256'].items():assert hashlib.sha256((j/'source-snapshot'/name).read_bytes()).hexdigest()==sha
assert hashlib.sha256((j/'residual-norm.metal').read_bytes()).hexdigest()==meta['shader_sha256']
assetfile=j/'norm-assets.json';assert hashlib.sha256(assetfile.read_bytes()).hexdigest()==meta['asset_manifest_sha256']
assets=json.loads(assetfile.read_text())['cases'];assert len(assets)==6
for a in assets:
 raw=(j/Path(a['path']).name).read_bytes();m=meta['assets'][a['model']+':'+a['tensor']];assert len(raw)==a['columns']*4 and hashlib.sha256(raw).hexdigest()==m['expanded_fp32_sha256']
 bits=[x[0] for x in struct.iter_unpack('<I',raw)];assert all(x&65535==0 for x in bits)
 canonical=b''.join(struct.pack('<H',x>>16) for x in bits);assert hashlib.sha256(canonical).hexdigest()==m['canonical_bf16_sha256']
stages=json.loads((j/'statuses.json').read_text());assert len(stages)==2 and all(s['exit_code']==0 for s in stages)
assert json.loads((j/'gpu-monitor.json').read_text())
rows=[json.loads(v) for v in (j/'gpu.log').read_text().splitlines() if v.startswith('{')]
checks=[r for r in rows if r['kind']=='check'];assert len(checks)==30
profiles=[]
for q in checks:
 assert q['bits_exact'] and q['padding_preserved'] and q['input_variants']==4 and 0<=q['fp64_worst_ratio']<=1 and q['validated_executions']>=328
 timing=[r for r in rows if r['kind']=='timing' and all(r[k]==q[k] for k in ('model','tensor','rows'))];assert len(timing)==12
 for r in timing:assert r['bits_exact'] and all(len(r[k])==24 for k in ('encode_ms','wall_ms','gpu_ms'))
 means={policy:{key:statistics.mean(statistics.median(r[key]) for r in timing if r['policy']==policy) for key in ('encode_ms','wall_ms','gpu_ms')} for policy in ('control','candidate')}
 rounds=[{key:statistics.mean(statistics.median(r[key]) for r in timing if r['policy']=='candidate' and r['round']==round)/statistics.mean(statistics.median(r[key]) for r in timing if r['policy']=='control' and r['round']==round) for key in ('encode_ms','wall_ms','gpu_ms')} for round in range(3)]
 profiles.append(dict(model=q['model'],tensor=q['tensor'],rows=q['rows'],means=means,ratios={key:means['candidate'][key]/means['control'][key] for key in means['control']},rounds=rounds))
result=dict(verified_files=len(files),profiles=profiles,exact_validated_executions=sum(q['validated_executions'] for q in checks),timing_windows=360,fp64_worst_ratio=max(q['fp64_worst_ratio'] for q in checks),full_model_gain_established=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
