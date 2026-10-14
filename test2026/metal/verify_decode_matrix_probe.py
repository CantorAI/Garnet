"""Verify actual FP32 matrix-decode component gates and independent timing ratios."""
import hashlib,json,statistics,sys
from pathlib import Path
base=Path(sys.argv[1]).resolve();e=base/'evidence';files=json.loads((e/'files.json').read_text())
for name,item in files.items():
 raw=(base/name).read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
job=e/'logs/decode-matrix-probe-20261014';assert json.loads((job/'exit.json').read_text())['exit_code']==0
j=job/'jobs';stages=json.loads((j/'statuses.json').read_text());assert len(stages)==8 and all(s['exit_code']==0 for s in stages)
metadata=json.loads((j/'metadata.json').read_text())
for name,sha in metadata['source_sha256'].items():assert hashlib.sha256((j/'source-snapshot'/name).read_bytes()).hexdigest()==sha
proof=json.loads((e/'omitted-matrix-proof.json').read_text());assert len(proof)==34
summary=json.loads((j/'summary.json').read_text());assert len(summary['checks'])==len(summary['results'])==136
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
 assets=j/(model+'-assets');manifest=json.loads((assets/'manifest.json').read_text());rows=[json.loads(v) for v in (j/(model+'-gpu.log')).read_text().splitlines() if v.startswith('{')]
 for case in manifest['cases']:
  key='logs/decode-matrix-probe-20261014/jobs/'+model+'-assets/'+case['name']+'.bin'
  assert proof[key]['sha256']==case['sha256'] and proof[key]['bytes']==case['columns']*case['inner']*2 and proof[key]['checkpoint_sha256']==manifest['checkpoint_sha256']
 checks=[r for r in rows if r['kind']=='check'];assert len(checks)==len(manifest['cases'])*4
 assert json.loads((j/(model+'-gpu-monitor.json')).read_text())
 for q in checks:
  assert q['variant'] in ('bf16_decode_matrix8','bf16_gemm') and q['padding_preserved']
  assert all(0<=q[k]<=1 for k in ('fp64_worst_ratio','control_fp64_worst_ratio','comparison_worst_ratio'))
  times=[r for r in rows if r['kind']=='timing' and all(r[k]==q[k] for k in ('matrix','rows','variant'))];assert len(times)==8
  control=statistics.median(r['gpu_ms_median'] for r in times if r['policy']=='control');candidate=statistics.median(r['gpu_ms_median'] for r in times if r['policy']=='candidate')
  result=next(r for r in summary['results'] if (r['model'],r['name'],r['rows'],r['variant'])==(model,q['matrix'],q['rows'],q['variant']))
  assert result['control_gpu_ms']==control and result['candidate_gpu_ms']==candidate and abs(result['component_speedup']-control/candidate)<1e-12
ratios={name:statistics.mean(r['candidate_gpu_ms'] for r in summary['results'] if r['variant']==name)/statistics.mean(r['control_gpu_ms'] for r in summary['results'] if r['variant']==name) for name in ('bf16_decode_matrix8','bf16_gemm')}
result=dict(verified_files=len(files),matrix_fixtures=34,reference_checks=136,timing_windows=1088,fp64_worst_ratio=max(q['fp64_worst_ratio'] for q in summary['checks']),pooled_gpu_fractions=ratios,full_model_gain_established=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
