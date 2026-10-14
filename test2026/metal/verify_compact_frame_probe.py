"""Verify exact preprocessing fixtures/real image bytes and CPU timing scope."""
import hashlib,json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();e=root/'evidence';files=json.loads((e/'files.json').read_text())
for name,proof in files.items():
 p=root/name;assert p.resolve().is_relative_to(root);raw=p.read_bytes();assert len(raw)==proof['bytes'] and hashlib.sha256(raw).hexdigest()==proof['sha256']
j=e/'logs/compact-frame-repair-probe-20261014';assert json.loads((j/'exit.json').read_text())['exit_code']==0
failed=e/'logs/compact-frame-probe-20261014';assert json.loads((failed/'exit.json').read_text())['exit_code']==1
assert "name 'output' is not defined" in (failed/'output.log').read_text()
summary=json.loads((j/'jobs/summary.json').read_text());compat=json.loads((j/'jobs/compatibility/verification.json').read_text())
assert compat['comparisons']==len(compat['checks'])==768 and compat['all_complete_bytes_match']
assert json.loads((j/'jobs/compatibility-status.json').read_text())['exit_code']==0
for name,sha in summary['source_sha256'].items():assert hashlib.sha256((j/'jobs/source-snapshot'/name).read_bytes()).hexdigest()==sha
records=summary['records'];assert len(records)==192 and len({r['rgb_sha256'] for r in records})==1
assert len(summary['warmups'])==2 and all(v['seconds']>=.25 and v['executions']>=2 for v in summary['warmups'])
for r in records:assert r['patch_sha256']==summary['cases'][str(r['batch'])]['image_metadata']['patch_sha256']
for batch in (1,2):
 for repeat in (0,1):
  windows=[r for r in records if r['batch']==batch and r['repeat']==repeat]
  for window,policy in enumerate((False,True,True,False)):
   values=[r for r in windows if r['window']==window];assert len(values)==12 and [r['sample'] for r in values]==list(range(12)) and all(r['candidate']==policy for r in values)
 for policy,key in ((False,'control_ms'),(True,'candidate_ms')):
  mean=statistics.mean(r['elapsed_ms'] for r in records if r['batch']==batch and r['candidate']==policy)
  result=next(v for v in summary['results'] if v['batch']==batch);assert abs(mean-result[key])<1e-12
 assert abs(result['fraction']-result['candidate_ms']/result['control_ms'])<1e-12
result=dict(verified_files=len(files),complete_compatibility_comparisons=768,exact_real_image_samples=192,results=summary['results'],scope='CPU preparation only; no image inference or new vLLM gain claim',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
