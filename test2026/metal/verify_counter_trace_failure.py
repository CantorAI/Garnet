"""Audit the capability success and preserved diagnostic channel failure."""
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();proof=json.loads((root/'manifest.json').read_text())
for e in proof['files']:
    p=root/e['path'];assert p.resolve().is_relative_to(root) and p.stat().st_size==e['bytes']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256']
cap=root/'logs/metal-counter-capabilities-20261015'
assert json.loads((cap/'exit.json').read_text())['exit_code']==0
hardware=json.loads((cap/'jobs/capabilities.log').read_text())
assert hardware['device']=='Apple M4' and hardware['stage_boundary'] and not hardware['dispatch_boundary'] and not hardware['blit_boundary']
assert hardware['sampled_compute_passes']==2 and hardware['exact_dependency_values']==4096
failed=root/'logs/counter-trace-profiles-20261015';assert json.loads((failed/'exit.json').read_text())['exit_code']==1
assert 'JSONDecodeError' in (failed/'output.log').read_text()
stages=json.loads((failed/'jobs/statuses.json').read_text())
assert len(stages)==6 and all(s['exit_code']==0 and s['interruption'] is None for s in stages)
assert [s['name'] for s in stages]==['generate','configure','build','ctest','Qwen3-0.6B-b1-quiet-0','Qwen3-0.6B-b1-traced-0']
for s in stages:assert json.loads((failed/'jobs'/(s['name']+'-monitor.json')).read_text())
assert '100% tests passed' in (failed/'jobs/ctest.log').read_text()
assert json.loads((failed/'jobs/metadata.json').read_text())['library_sha256']==proof['candidate_sha256']
profiles=json.loads((failed/'jobs/profiles.json').read_text())['profiles'];assert len(profiles)==1
p=profiles[0];assert not p['tracing'] and p['requests']==3 and p['complete_outputs_match']
expected=json.loads((failed/'jobs/Qwen3-0.6B-b1-expected.json').read_text())
assert len(p['records'])==3 and all(all(r[k]==expected[k] for k in ('token_ids','prompt_token_ids','finished')) for r in p['records'])
bad=(failed/'jobs/Qwen3-0.6B-b1-traced-0.log').read_text().splitlines()
malformed=0
for line in bad:
    if line.startswith('matched_result '):
        try:json.loads(line.split(' ',1)[1])
        except json.JSONDecodeError:
            assert 'metal_counter_' in line;malformed+=1
assert malformed>0
result=dict(verified_files=len(proof['files']),capability_exact_values=4096,numeric_ctest_passed=True,quiet_complete_output_requests=3,
    traced_worker_exit_zero_but_outputs_unverified=True,malformed_result_lines=malformed,
    reason='GPU trace and result streams interleaved; strict JSON parser rejected the run',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
