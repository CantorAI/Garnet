"""Regenerate exact private context and verify primitive compile/controls/guard evidence."""
from pathlib import Path
import hashlib,json,runpy,sys
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text());trial='prepared-union-mlp-primitive-20261016'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    p=(base/name).resolve();assert p.is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
assert proof['jobs'][trial]['exit_code']==0 and 'Apple M4' in proof['hardware']['machdep.cpu.brand_string'] and int(proof['hardware']['hw.memsize'])==16*2**30
jobs=root/'logs'/trial/'jobs';meta=json.loads((jobs/'metadata.json').read_text());snapshots={}
for i,(name,sha) in enumerate(meta['source_sha256'].items()):
    p=jobs/'source-snapshot'/str(i)/Path(name).name;assert digest(p)==sha and p.name not in snapshots;snapshots[p.name]=p
factory=runpy.run_path(str(snapshots['make_prepared_union_mlp.py']))
for name,operation in (('metal_context.h','header'),('metal_context.mm','context')):
    p=jobs/'generated'/name;assert p.read_bytes()==factory[operation](snapshots[name].read_text()).encode() and digest(p)==meta['generated_sha256'][name]
assert meta['reference_library_sha256']=='d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e'
assert digest(snapshots['exact_half_codec.h'])=='e8800f4983bb2786a7e376ca7784a7817176a5e6dd2d46c7e353116f99b2a4c9'
assert '-DGARNET_METAL_GLOBAL_HALF_ALPHA=1' in meta['defines']
stages=json.loads((jobs/'statuses.json').read_text());assert [s['name'] for s in stages]==['compile','primitive'] and all(s['exit_code']==0 for s in stages)
for stage in stages:
    monitor=json.loads((jobs/(stage['name']+'-monitor.json')).read_text());assert monitor and all(r['guard_passed'] for r in monitor)
compile=stages[0]['command'];assert all(d in compile for d in meta['defines']) and '-O3' in compile and '-fobjc-arc' in compile
assert (jobs/'primitive.log').read_text().strip()=='Prepared union native unequal-scale/dyadic24/pending/ownership/alias/abort-recovery/fallback/cache128-reject2 PASS weight_bytes=512'
assert proof['omitted']['logs/'+trial+'/jobs/prepared_union_mlp_primitive_probe']['sha256']==meta['binary_sha256']
result=dict(verified_files=len(proof['files']),original_dyadic_cases=24,cache_admitted_pairs=128,cache_rejected_pairs=2,initial_prepared_weight_buffer_bytes=512,
 regenerated_context_and_header=True,pending_foreign_alias_abort_recovery_controls=True,
 scope='Captured private synthetic GPU primitive only; no learned matrices, native graph integration, memory saving or model speed claim.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
