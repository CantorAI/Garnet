"""Regenerate and audit private canonical output-view component; no native consumer claim."""
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
factory=runpy.run_path(str(snapshots['make_prepared_union_qv_views.py']))
for name,operation in [('metal_context.h','header'),('metal_context.mm','context')]:
 p=jobs/'generated'/name;assert p.read_bytes()==factory[operation](snapshots[name].read_text()).encode() and digest(p)==meta['generated_sha256'][name]
context=(jobs/'generated/metal_context.mm').read_text();body=context[context.index('bool MetalContext::EncodePreparedUnionQvViewsOriginalK('):context.index('size_t MetalContext::LastPreparedUnionQvViewsOriginalKDispatches()')]
assert '[item.product encodeToCommandBuffer:' in body and 'EncodeBF16MatVec(input,k,kOut);' in body and 'computeCommandEncoder' not in body and 'preparedQvViewsOriginalKScatter' not in body
assert 'matrix.data.contents)+matrix.offset+i*matrix.rowBytes' in context
assert meta['reference_library_sha256']=='d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e'
stages=json.loads((jobs/'statuses.json').read_text());assert [v['name'] for v in stages]==['compile','primitive'] and all(v['exit_code']==0 for v in stages)
for stage in stages:
 monitor=json.loads((jobs/(stage['name']+'-monitor.json')).read_text());assert monitor and all(v['guard_passed'] for v in monitor)
assert (jobs/'primitive.log').read_text().strip()=='Prepared QV views24/original-key/canonical-offset/guard/pending/foreign/abort-recovery/fallback/cache64-reject2 PASS weight_bytes=256'
assert proof['omitted']['logs/'+trial+'/jobs/prepared_union_qv_views_primitive_probe']['sha256']==meta['binary_sha256']
result=dict(verified_files=len(proof['files']),dyadic_cases=24,cache_admitted_groups=64,cache_rejected_groups=2,initial_extra_union_weight_bytes=256,regenerated_context=True,original_key_bits_checked=True,canonical_view_offsets_and_guard_checked=True,no_scatter_encode_in_candidate=True,downstream_gpu_consumers_adapted=False,scope='Private component outputs read only through offset-aware Download. GPU consumers, host writes and graph/session integration are unimplemented; no native speed admission.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
