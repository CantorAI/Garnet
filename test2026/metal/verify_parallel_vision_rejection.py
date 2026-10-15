"""Independent rejection audit: original gates passed, full-grid tolerance failed."""
import hashlib,json,math,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();proof=json.loads((root/'evidence/manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,v in proof['files'].items():
    p=root/name;assert p.resolve().is_relative_to(root) and p.stat().st_size==v['bytes'] and digest(p)==v['sha256']
for name,sha in proof['installed'].items():assert digest(root/'evidence/source'/name)==sha
for name,sha in proof['original'].items():assert digest(root/'evidence/logs/parallel-vision-original-source-20261015'/name)==sha
assert proof['control_sha256']=='887b70f521d5af7f948bcb32a2cf94842377e2ecb828bc9280ffb38af6fe058f'
assert proof['jobs']['parallel-vision-checks-20261015']['exit_code']==1 and proof['jobs']['parallel-vision-native-20261015']['exit_code']==1
out=root/'evidence/logs/parallel-vision-checks-20261015/jobs';meta=json.loads((out/'metadata.json').read_text());assert meta['candidate_sha256']==proof['candidate_sha256']
for name,sha in meta['source_sha256'].items():assert digest(root/'evidence/source'/name)==sha
stages=json.loads((out/'statuses.json').read_text());assert len(stages)==19 and len({s['name'] for s in stages})==19 and all(s['exit_code']==0 for s in stages)
for stage in stages:
    assert stage['accepted_cases']
    for c in stage['accepted_cases']:
        folder=out/c['evidence'];statuses=json.loads((folder/'statuses.json').read_text());assert statuses and all(s['exit_code']==0 for s in statuses)
        assert json.loads((folder/(stage['name']+'-monitor.json')).read_text())
failure=out/'attempts/vision-full-grid/case0-attempt0';text=(failure/'vision-full-grid.log').read_text()
assert 'trained full-grid fresh encoding and counters PASS 1 False [0, 0]' in text and 'trained full-grid fresh encoding and counters PASS 1 True [12, 12]' in text
assert 'AssertionError' in text and 'worst_ratio' in text
full=out/'full-grid';control=json.loads((full/'control-b1.json').read_text());candidate=json.loads((full/'candidate-b1.json').read_text());fixture=json.loads((full/'candidate-b1-fixture.json').read_text())
a=control['values'];b=candidate['values'];assert len(a)==len(b)==fixture['config']['out_hidden_size']*fixture['grid_h']*fixture['grid_w']//fixture['config']['spatial_merge_size']**2
assert not control['enabled'] and candidate['enabled'] and control['counts']==[0,0] and candidate['counts']==[12,12]
assert all(math.isfinite(x) for x in a+b);errors=[abs(x-y) for x,y in zip(a,b)];ratios=[d/(5e-6+5e-5*abs(x)) for d,x in zip(errors,a)]
assert max(ratios)>1 and not (full/'candidate-b2.json').exists()
assert 'bound=5e-6+5e-5*np.abs(a)' in (root/'evidence/source/test2026/metal/test_parallel_vision_grid.py').read_text()
native=root/'evidence/logs/parallel-vision-native-20261015';assert not (native/'native').exists()
result=dict(verified_files=len(proof['files']),original_successful_stages=19,full_grid_batch=1,embedding_values=len(a),actual_candidate_dispatches=candidate['counts'],max_abs_error=max(errors),worst_tolerance_ratio=max(ratios),failed_values=sum(r>1 for r in ratios),native_requests=0,candidate_sha256=proof['candidate_sha256'],decision='Rejected under unchanged trained full-grid bound. B2 and native performance did not execute.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
