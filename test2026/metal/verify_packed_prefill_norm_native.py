"""Audit all original validation cases, full outputs and actual prefill counters."""
import hashlib,json,math,statistics,sys,tarfile
from pathlib import Path
base=Path(sys.argv[1]).resolve();proof=json.loads((base/'manifest.json').read_text())
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for e in proof['files']:
 p=base/e['path'];assert p.resolve().is_relative_to(base);assert p.stat().st_size==e['bytes'] and digest(p)==e['sha256']
assert proof['jobs']['packed-prefill-norm-resume-checks-20261014']['exit_code']==proof['jobs']['packed-prefill-norm-resume-native-20261014']['exit_code']==0
assert proof['jobs']['packed-prefill-norm-checks-20261014']['exit_code']==proof['jobs']['packed-prefill-norm-native-20261014']['exit_code']==1
staging=base/'packed-prefill-norm-staging';manifest=json.loads((staging/'manifest.json').read_text())
with tarfile.open(staging/'source.tar.gz') as tar:
 assert {m.name for m in tar.getmembers()}==set(manifest)
 for m in tar.getmembers():assert m.isfile() and hashlib.sha256(tar.extractfile(m).read()).hexdigest()==manifest[m.name]==digest(base/'source'/m.name)
original=json.loads((staging/'original.json').read_text())
for name,sha in original.items():assert digest(base/'logs/packed-prefill-norm-original-source-20261014'/name)==sha
checks=base/'logs/packed-prefill-norm-resume-checks-20261014/jobs';stages=json.loads((checks/'statuses.json').read_text());assert len(stages)==19
assert all(v['exit_code']==0 and v['interruption'] is None for v in stages)
prior=base/'logs/packed-prefill-norm-checks-20261014/jobs';old=json.loads((prior/'statuses.json').read_text());assert stages[:3]==old[:3]
assert 'Another inference runtime is active' in old[-1]['interruption']
for stage in stages[3:]:
 accepted=stage['accepted_cases'];assert [v['case_index'] for v in accepted]==list(range(8 if stage['name'].endswith('-exact-tails') else 1))
 for case in accepted:
  folder=checks/case['evidence'];assert folder.resolve().is_relative_to(checks)
  records=json.loads((folder/'statuses.json').read_text());assert len(records)==1 and records[0]==case['status'] and records[0]['exit_code']==0 and records[0]['interruption'] is None
  assert json.loads((folder/(stage['name']+'-monitor.json')).read_text())
  if stage['name'].endswith('-exact-tails'):
   assert case['command'][1].endswith('test_exact_dense_tail_case_reference.py') and int(case['command'][-1])==case['case_index']
   assert 'Original dense-tail fixture passed case_index='+str(case['case_index']) in (folder/(stage['name']+'.log')).read_text()
meta=json.loads((checks/'metadata.json').read_text());assert meta['candidate_sha256']==proof['candidate_sha256']
assert 'METAL_PACKED_PREFILL_NORM' in meta['options'] and all(v not in meta['options'] for v in ('METAL_DIRECT_STATE_ZERO','METAL_CONSTANT_BF16_HEAD','METAL_JOINED_PREFILL','METAL_RESIDUAL_NORM'))
for name,sha in meta['source_sha256'].items():assert digest(base/'source'/name)==sha
folder=base/'logs/packed-prefill-norm-resume-native-20261014/jobs';outer=json.loads((folder/'summary.json').read_text())
assert {(p['model'],p['batch']) for p in outer['profiles']}=={(m,b) for m in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B') for b in (1,2)}
profiles=[];count=0
for p in outer['profiles']:
 label=p['model']+'-b'+str(p['batch']);where=folder/Path(p['evidence']).name;summary=json.loads((where/'summary.json').read_text());records=summary['records'];assert len(records)==12
 reference=json.loads((where/(label+'-expected.json')).read_text());metadata=json.loads((where/'metadata.json').read_text())
 assert metadata['library_sha256']=={'control':proof['control_sha256'],'candidate':proof['candidate_sha256']}
 layers=28 if p['model']=='Qwen3-0.6B' else 24
 for r in records:
  assert all(r[k]==reference[k] for k in ('token_ids','prompt_token_ids','finished')) and r['case']==records[0]['case'] and r['case']['batch']==p['batch']
  candidate=r['build_policy']=='candidate';assert r['packed_prefill_norm_enabled']==candidate
  assert r['prefill_profile'].get('packed_prefill_norm_dispatches',0)==(layers*len(r['prefill_chunk_plan']) if candidate else 0)
  assert r['attention_prefix_enabled'] and not r['constant_head_enabled'] and not r['direct_state_zero_enabled'] and not r['residual_norm_enabled']
  if p['model']=='Qwen3.5-0.8B':assert r['vision_reencoded'] and r['patch_sha256']==r['case']['image_metadata']['patch_sha256']
 extra={}
 for policy in ('control','candidate'):
  warm=[r for r in records if r['build_policy']==policy and r['request']>0];assert len(warm)==4
  for key,field,scale in [('ttft_ms','ttft_seconds',1000),('decode_tps','decode_wall_steps_per_second',1),('initialization_s','initialization_seconds',1),('peak_individual_rss_bytes','peak_individual_rss_bytes',1)]:assert math.isclose(p['means'][policy][key],statistics.mean(r[field] for r in warm)*scale,rel_tol=1e-12)
  first=[r for r in records if r['build_policy']==policy and r['request']==0];assert len(first)==2
  extra[policy]=dict(first_request_after_init_ttft_ms=statistics.mean(r['ttft_seconds'] for r in first)*1000,aggregate_decode_tps=p['batch']*p['means'][policy]['decode_tps'])
  for repeat in (0,1):
   worker=where/(label+'-'+policy+'-'+str(repeat));assert json.loads((worker/'exit.json').read_text())['exit_code']==0 and json.loads((worker/'monitor.json').read_text())
 assert math.isclose(p['decode_ratio'],p['means']['candidate']['decode_tps']/p['means']['control']['decode_tps'],rel_tol=1e-12) and math.isclose(p['ttft_fraction'],p['means']['candidate']['ttft_ms']/p['means']['control']['ttft_ms'],rel_tol=1e-12)
 profiles.append(dict(**p,distinct_timing_phases=extra));count+=12
result=dict(verified_files=len(proof['files']),numerical_stages=19,full_output_requests=count,all_actual_prefill_counters_match=True,profiles=profiles,scope='Packed prefill versus identical attention; no fresh vLLM comparison',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k!='profiles'}))
