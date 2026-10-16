"""Regenerate heterogeneous-QV with original K context and audit actual primitive/compile/guard records."""
from pathlib import Path
import hashlib,json,runpy,sys
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
 p=(base/name).resolve();assert p.is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
assert len(proof['jobs'])==1;trial,status=next(iter(proof['jobs'].items()));assert status['exit_code']==0
assert 'Apple M4' in proof['hardware']['machdep.cpu.brand_string'] and int(proof['hardware']['hw.memsize'])==16*2**30
jobs=root/'logs'/trial/'jobs';meta=json.loads((jobs/'metadata.json').read_text());snapshots={}
for i,(name,sha) in enumerate(meta['source_sha256'].items()):
 p=jobs/'source-snapshot'/str(i)/Path(name).name;assert digest(p)==sha and p.name not in snapshots;snapshots[p.name]=p
factory=runpy.run_path(str(snapshots['make_prepared_union_qv.py']))
for name,operation in [('metal_context.h','header'),('metal_context.mm','context')]:
 p=jobs/'generated'/name;assert p.read_bytes()==factory[operation](snapshots[name].read_text()).encode() and digest(p)==meta['generated_sha256'][name]
assert meta['reference_library_sha256']=='d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e'
assert digest(snapshots['exact_half_codec.h'])=='e8800f4983bb2786a7e376ca7784a7817176a5e6dd2d46c7e353116f99b2a4c9'
stages=json.loads((jobs/'statuses.json').read_text());assert [v['name'] for v in stages]==['compile','primitive'] and all(v['exit_code']==0 for v in stages)
for stage in stages:
 monitor=json.loads((jobs/(stage['name']+'-monitor.json')).read_text());assert monitor and all(v['guard_passed'] for v in monitor)
assert all(v in stages[0]['command'] for v in meta['defines']) and '-O3' in stages[0]['command'] and '-fobjc-arc' in stages[0]['command']
assert (jobs/'primitive.log').read_text().strip()=='Prepared QV with original K heterogeneous24/fp32-key/pending/ownership/alias/abort-recovery/fallback/cache64-reject2 PASS weight_bytes=256'
assert proof['omitted']['logs/'+trial+'/jobs/prepared_union_qv_primitive_probe']['sha256']==meta['binary_sha256']
result=dict(verified_files=len(proof['files']),dyadic_cases=24,cache_admitted_groups=64,cache_rejected_groups=2,initial_extra_union_weight_bytes=256,regenerated_context=True,original_fp32_key_fixture=True,logical_outputs_and_joined_scratch_checked=True,pending_foreign_alias_abort_recovery_controls=True,scope='Synthetic actual-context primitive only; no physical-padding/captured-activation/learned arithmetic or model speed admission.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
