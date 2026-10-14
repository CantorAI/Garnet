"""Verify actual guarded checks, complete outputs, route counters and warm clocks."""
import hashlib,json,math,statistics,sys
from pathlib import Path
base=Path(sys.argv[1]).resolve();proof=json.loads((base/'manifest.json').read_text())
for e in proof['files']:
 p=base/e['path'];assert p.resolve().is_relative_to(base)
 assert p.stat().st_size==e['bytes'] and hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==e['sha256']
assert all(x['exit_code']==0 for x in proof['jobs'].values())
checks=base/'logs/constant-head-checks-20261014/jobs';stages=json.loads((checks/'statuses.json').read_text())
assert len(stages)==19 and all(x['exit_code']==0 and x['interruption'] is None for x in stages)
metadata=json.loads((checks/'metadata.json').read_text());assert metadata['candidate_sha256']==proof['candidate_sha256']
assert 'METAL_CONSTANT_BF16_HEAD' in metadata['options'] and 'METAL_DIRECT_STATE_ZERO' not in metadata['options']
folder=base/'logs/constant-head-native-20261014/jobs';outer=json.loads((folder/'summary.json').read_text());assert len(outer['profiles'])==3
profiles=[];count=0
for p in outer['profiles']:
 assert p['batch']==1
 where=folder/Path(p['evidence']).name;summary=json.loads((where/'summary.json').read_text());records=summary['records'];assert len(records)==12
 reference=json.loads((where/(p['model']+'-b1-expected.json')).read_text())
 meta=json.loads((where/'metadata.json').read_text())
 assert meta['library_sha256']=={'control':proof['control_sha256'],'candidate':proof['candidate_sha256']}
 for r in records:
  assert all(r[k]==reference[k] for k in ('token_ids','prompt_token_ids','finished'))
  assert r['case']==records[0]['case']
  candidate=r['build_policy']=='candidate';assert r['constant_head_enabled']==candidate and r['constant_head_dispatches']==int(candidate)
  assert r['attention_prefix_enabled']
  if p['model']=='Qwen3.5-0.8B':assert r['vision_reencoded'] and r['patch_sha256']==r['case']['image_metadata']['patch_sha256']
 for policy in ('control','candidate'):
  warm=[r for r in records if r['build_policy']==policy and r['request']>0];assert len(warm)==4
  for key,field,scale in [('ttft_ms','ttft_seconds',1000),('decode_tps','decode_wall_steps_per_second',1),('initialization_s','initialization_seconds',1),('peak_individual_rss_bytes','peak_individual_rss_bytes',1)]:
   assert math.isclose(p['means'][policy][key],statistics.mean(r[field] for r in warm)*scale,rel_tol=1e-12)
  for repeat in (0,1):
   worker=where/(p['model']+'-b1-'+policy+'-'+str(repeat));assert json.loads((worker/'exit.json').read_text())['exit_code']==0
   assert json.loads((worker/'monitor.json').read_text())
 assert math.isclose(p['decode_ratio'],p['means']['candidate']['decode_tps']/p['means']['control']['decode_tps'],rel_tol=1e-12)
 assert math.isclose(p['ttft_fraction'],p['means']['candidate']['ttft_ms']/p['means']['control']['ttft_ms'],rel_tol=1e-12)
 profiles.append(p);count+=12
result=dict(verified_files=len(proof['files']),numerical_stages=19,full_output_requests=count,all_route_counters_match=True,profiles=profiles,full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
