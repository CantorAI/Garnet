"""Audit actual paired norm-RoPE original-bit primitive and regenerate exact arithmetic body."""
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
factory=runpy.run_path(str(snapshots['make_paired_norm_rope.py']))
for name,operation in [('metal_context.h','header'),('metal_context.mm','context')]:
 p=jobs/'generated'/name;assert p.read_bytes()==factory[operation](snapshots[name].read_text()).encode() and digest(p)==meta['generated_sha256'][name]
original=snapshots['metal_context.mm'].read_text();start=original.index('kernel void norm_rope(');end=original.index('\nstruct KVParams ',start);kernel=original[start:end];body=kernel[kernel.index('    if(id>='):]
assert factory['shader'](original).endswith(body+'\n') and 'simd_sum(sum)' in body and 'precise::cos' in body and 'precise::sin' in body
assert meta['reference_library_sha256']=='d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e'
stages=json.loads((jobs/'statuses.json').read_text());assert [v['name'] for v in stages]==['compile','primitive'] and all(v['exit_code']==0 for v in stages)
for stage in stages:
 monitor=json.loads((jobs/(stage['name']+'-monitor.json')).read_text());assert monitor and all(v['guard_passed'] for v in monitor)
assert all(v in stages[0]['command'] for v in meta['defines']) and '-O3' in stages[0]['command'] and '-fobjc-arc' in stages[0]['command']
assert (jobs/'primitive.log').read_text().strip()=='Paired norm-RoPE60/original-bits/pending/foreign/alias/abort-recovery/fallback PASS checked_values=3710720'
assert proof['omitted']['logs/'+trial+'/jobs/paired_norm_rope_primitive_probe']['sha256']==meta['binary_sha256']
result=dict(verified_files=len(proof['files']),synthetic_changed_input_cases=60,original_fp32_output_bits_checked=3710720,regenerated_context_and_original_arithmetic_body=True,original_32_lane_reduction_and_precise_trig=True,pending_foreign_alias_abort_recovery_fallback_checks=True,learned_weights_tested=False,timing_tested=False,native_integration=False,scope='Synthetic actual private-context bit identity/ownership prerequisite only. No learned/model accuracy or speed claim.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
