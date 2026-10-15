"""Audit all original native gates plus the graph primitive from frozen evidence."""
from pathlib import Path
import sys

# Preserve the original checks' exact numerical, trained-head, image and
# recovery assertions. Only archive framing/build identities change here.
p=Path(__file__).with_name('verify_coalesced_body_native.py')
wrapper=p.read_text();ns={'__file__':str(p)}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],str(p),'exec'),ns)
old=ns['source']
prefix=old[old.index("assert 'METAL_INTERLEAVED_SCREEN_INDEX'"):old.index("trial=root/'logs'/native_trial")]
recovery=old[old.index('recovered=0;prefix_recovered=0'):old.index('if confirmation_trial:')]
wide=old[old.index("stage=next(v for v in json.loads((checks/'statuses.json')"):old.index('for p in profiles:',old.index("stage=next(v for v in json.loads((checks/'statuses.json')"))]
source='''import hashlib,json,sys,re,math
from pathlib import Path
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    p=base/name;assert p.resolve().is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
trial='mlp-prefill-native-checks-20261015';assert proof['jobs'][trial]['exit_code']==0
checks=root/'logs'/trial/'jobs';meta=json.loads((checks/'metadata.json').read_text())
assert meta['candidate_sha256']==proof['omitted']['build-mlp-prefill-graph/libgarnet.dylib']['sha256']
assert proof['omitted']['build-coalesced-bf16-body/libgarnet.dylib']['sha256']=='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1'
manifest=json.loads((root/'mlp-prefill-native-staging/manifest.json').read_text())
assert len(manifest)==17
for name,sha in manifest.items():assert digest(root/'source'/name)==sha
assert 'METAL_MLP_PREFILL_GRAPH' in meta['options'] and 'METAL_COALESCED_BF16_BODY' in meta['options']
''' + prefix + recovery + wide + '''
assert 'metal_mlp_prefill_graph_reference' in ctext and '0 tests failed out of 8' in ctext
last=(checks/'CTest-LastTest.log').read_text()
assert 'MLP graph pending producer, ownership, abort/recovery and native bound PASS: 64 cases' in last
for name in ('metal_wide_batch_prefill_reference','metal_interleaved_bf16_body_reference','metal_coalesced_bf16_body_reference'):
    assert name in ctext
result=dict(verified_files=len(proof['files']),validation_stages=28,ctests=8,graph_primitive_cases=64,
    trained_head_queries=51,trained_head_rows=78,vision_grid_profiles=4,
    screened_recovery_cases=recovered,prefix_recovery_cases=prefix_recovered,
    wide_reference_profiles=4,wide_carried_steps=12,worst_wide_original_bound_ratio=max(ratios),
    candidate_library_sha256=meta['candidate_sha256'],source_manifest=manifest,
    scope='Native correctness/ownership checks only; no complete inference comparison, speed or vLLM claim',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\\n').encode())
print(json.dumps({k:v for k,v in result.items() if k!='source_manifest'}))
'''
exec(compile(source,__file__,'exec'))
