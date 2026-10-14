"""Audit the preserved successful gates and incomplete rejected timing profile."""
import hashlib,json,sys,tarfile
from pathlib import Path
base=Path(sys.argv[1]).resolve();proof=json.loads((base/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for info in proof['files']:
 p=base/info['path'];assert p.resolve().is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
assert proof['candidate_sha256']=='e389fc080249301b506b84ff09b242df7d6cd63128091e1ab677b4025710491c' and proof['control_sha256']=='887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f'
assert proof['jobs']['dirty-prefix-reset-checks-20261014']['exit_code']==0 and proof['jobs']['dirty-prefix-reset-native-20261014']['exit_code']==1
stage=base/'dirty-prefix-reset-staging';manifest=json.loads((stage/'manifest.json').read_text())
with tarfile.open(stage/'source.tar.gz') as archive:
 assert {m.name for m in archive.getmembers()}==set(manifest)
 for m in archive.getmembers():assert hashlib.sha256(archive.extractfile(m).read()).hexdigest()==manifest[m.name]==digest(base/'source'/m.name)
for name,sha in json.loads((stage/'original.json').read_text()).items():assert digest(base/'logs/dirty-prefix-reset-original-source-20261014'/name)==sha
checks=base/'logs/dirty-prefix-reset-checks-20261014/jobs';stages=json.loads((checks/'statuses.json').read_text());assert len(stages)==19 and all(s['exit_code']==0 and s['interruption'] is None for s in stages)
for s in stages:
 accepted=s['accepted_cases'];assert [v['case_index'] for v in accepted]==list(range(8 if s['name'].endswith('-exact-tails') else 1))
 for case in accepted:
  folder=checks/case['evidence'];assert json.loads((folder/'statuses.json').read_text())==[case['status']] and case['status']['exit_code']==0
  assert json.loads((folder/(s['name']+'-monitor.json')).read_text())
  if s['name'].endswith('-exact-tails'):assert 'Original dense-tail fixture passed case_index='+str(case['case_index']) in (folder/(s['name']+'.log')).read_text()
assert 'Dirty prefix reset complete CPU/GPU positive zero, bounds, uninitialized and abort PASS 216' in (checks/'CTest-LastTest.log').read_text()
gates=base/'logs/dirty-prefix-reset-native-20261014/jobs-session-gates';statuses=json.loads((gates/'statuses.json').read_text());assert len(statuses)==3 and all(v['exit_code']==0 for v in statuses)
for s in statuses:
 values=[json.loads(line.split(' ',2)[2]) for line in (gates/(s['name']+'.log')).read_text().splitlines() if line.startswith('prefix_reset_recovery PASS ')];assert len(values)==2 and {v['batch'] for v in values}=={1,2}
 assert all(all(v[k] for k in ('partial_clear_verified','unknown_writer_full_fallback','nonfinite_request_poisoned','full_recovery_verified','private_sessions_isolated')) for v in values)
folder=base/'logs/dirty-prefix-reset-native-20261014/jobs';assert 'line 101' in (folder/'Qwen3-0.6B-b1-attempt0.log').read_text() and 'AssertionError' in (folder/'Qwen3-0.6B-b1-attempt0.log').read_text()
profile=folder/'Qwen3-0.6B-b1-attempt0';expected=json.loads((profile/'Qwen3-0.6B-b1-expected.json').read_text());outputs=0
for policy in ('control','candidate'):
 worker=profile/('Qwen3-0.6B-b1-'+policy+'-0');assert json.loads((worker/'exit.json').read_text())['exit_code']==0
 rows=[json.loads(line.split(' ',1)[1]) for line in (worker/'output.log').read_text().splitlines() if line.startswith('matched_result ')];assert len(rows)==3
 for r in rows:assert all(r[k]==expected[k] for k in ('token_ids','prompt_token_ids','finished'))
 if policy=='candidate':
  assert rows[0]['dirty_prefix_reset_states']==56 and rows[0]['dirty_prefix_reset_bytes']==0
  assert all(r['dirty_prefix_reset_states']==0 and r['dirty_prefix_reset_fallback_states']==56 and r['dirty_prefix_reset_bytes']==58720256 for r in rows[1:])
 outputs+=len(rows)
result=dict(verified_files=len(proof['files']),original_validation_stages=19,session_recovery_checks=6,rejected_profile_matching_outputs=outputs,reason='Prefill reshape copy rejected admission; both warm candidate requests used full reset',scope='Incomplete rejected timing profile; no serving performance claim',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
