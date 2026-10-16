"""Audit a numerical rejection without weakening bounds or calling an incomplete suite successful."""
from pathlib import Path
import hashlib,json,math,re,sys
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
 p=(base/name).resolve();assert p.is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
assert proof['prior_archive_sha256']=='544e11d2c2f18582ceffb6a12404fb3113d4f03d5b382b55944076894e849c1a'
old='prepared-qkv-native-checks-20261016';new='prepared-qkv-native-checks-countfix-20261016';diag='prepared-qkv-exact-control-diagnostic-20261016'
assert proof['jobs'][old]['exit_code']==proof['jobs'][new]['exit_code']==proof['jobs']['prepared-qkv-native-finalize-countfix-20261016']['exit_code']==1 and proof['jobs'][diag]['exit_code']==0
prior=root/'logs'/old/'jobs';checks=root/'logs'/new/'jobs';oldstages=json.loads((prior/'statuses.json').read_text());stages=json.loads((checks/'statuses.json').read_text());assert len(oldstages)==25 and len(stages)==31 and stages[:25]==oldstages and all(v['exit_code']==0 for v in stages)
resume=json.loads((checks/'resume-provenance.json').read_text());assert len(resume['reused_stages'])==25 and resume['original_count_assertions']==[23,29,33] and resume['corrected_counts']==[25,31,33]
meta=json.loads((checks/'metadata.json').read_text());assert meta['candidate_sha256']==proof['library_sha256']['build-prepared-qkv-text']=='e34fc562b4840af94eed5e4a6043cdf912a4beb15a5c48d75d74ef7946012f77'
candidate_path=checks/'attempts/prepared-qkv-trained-reference/case0-attempt0/prepared-qkv-trained-reference.log';candidate=candidate_path.read_text();assert 'AssertionError' in candidate and 'key_rotation_diagnostic' in candidate
ratios=[float(v) for v in re.findall(r' worst_ratio ([0-9.eE+-]+)',candidate)];assert len(ratios)==10 and all(math.isfinite(v) for v in ratios) and max(ratios)==1.5165457144345686 and ratios[-1]>1
routes=[json.loads(line.split(' ',1)[1]) for line in candidate.splitlines() if line.startswith('prepared_qkv_trained_route ')];assert len(routes)==10 and routes[-1]==dict(batch=2,chunk=22,expected=1,session=1,graph=1)
diagnostic=root/'logs'/diag/'jobs';dmeta=json.loads((diagnostic/'metadata.json').read_text());fixture=checks/'trained-qkv-reference/layer0-b2-chunk22/fixture.json';assert digest(fixture)==dmeta['fixture_sha256']
for name,sha in dmeta['input_sha256'].items():assert proof['omitted'][name]['sha256']==sha
assert dmeta['control_library_sha256']==proof['library_sha256']['build-vector-vision-width4']=='d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e'
assert digest(root/'control-source/run_dense_reference.py')==dmeta['control_driver_sha256']
control=(diagnostic/'attempts/accepted-d52-exact-trained-fixture/case0-attempt0/accepted-d52-exact-trained-fixture.log').read_text();observed=[float(v) for v in re.findall(r' worst_ratio ([0-9.eE+-]+)',control)]
assert len(observed)==3 and max(observed)==0.5204084868377323 and all(0<=v<=1 for v in observed) and 'pretrained dense PASS full_attention 2 22' in control
done=json.loads((diagnostic/'statuses.json').read_text());assert len(done)==1 and done[0]['exit_code']==0
result=dict(verified_files=len(proof['files']),reused_passing_stages=25,fresh_passing_recovery_stages=6,total_passing_stages_before_rejection=31,candidate_failed_original_bound_ratio=max(ratios),accepted_control_worst_original_bound_ratio=max(observed),exact_saved_fixture_inputs=len(dmeta['input_sha256']),control_reference_steps=3,candidate_rejected=True,native_performance_admitted=False,scope='Full-QKV native path fails original trained key-cache bound while accepted d52 passes the exact saved fixture. No tolerance change, model benchmark or promotion.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
