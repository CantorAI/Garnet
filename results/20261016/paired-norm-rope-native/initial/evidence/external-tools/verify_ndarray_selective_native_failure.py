"""Audit preserved packed-layout compatibility failure, not full validation or performance."""
from pathlib import Path
import hashlib,json,runpy,sys
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    p=(base/name).resolve();assert p.is_relative_to(base)
    assert p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
assert json.loads((base/'archive-verification.json').read_text())['archive_sha256']=='a04efb9061130bb856a399168e23967f5ecedd6a9254632507a20508b15dbf68'
trial='ndarray-selective-native-checks-20261016';assert proof['jobs'][trial]['exit_code']==1
prep=json.loads((root/'preparation.json').read_text());checks=root/'logs'/trial/'jobs'
assert prep['baseline_library_sha256']=='c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1'
for name,sha in prep['source_sha256'].items():
    if 'evidence/source/'+name in proof['files']:assert digest(root/'source'/name)==sha
sys.path.insert(0,str(root/'source/test2026/metal'))
factory=runpy.run_path(str(root/'source/test2026/metal/make_ndarray_selective_native.py'))
operations={'src/backends/metal/metal_context.mm':'context','src/backends/metal/metal_context.h':'header',
    'src/backends/metal/decoder_session.cpp':'session','src/backends/metal/decoder_session.h':'session_header',
    'src/entry/metal_main.cpp':'entry','xModel/qwen35/vl_0_8b/metal_decoder.py':'decoder',
    'test2026/metal/run_matched_garnet.py':'worker','test2026/metal/frame_worker_garnet.py':'worker'}
assert set(prep['changes'])==set(operations)
for name,operation in operations.items():
    original=root/'original-source'/name
    assert digest(original)==prep['changes'][name]['before']
    assert factory[operation](original.read_text()).encode()==(root/'source'/name).read_bytes()
    assert digest(root/'source'/name)==prep['changes'][name]['after']
stages=json.loads((checks/'statuses.json').read_text())
assert [s['name'] for s in stages]==['configure','build','ndarray-cache-compile','ndarray-cache']
for stage in stages:
    assert stage['exit_code']==0
    for case in stage['accepted_cases']:
        assert case['status']['exit_code']==0 and case['status']['interruption'] is None
        monitors=list((checks/case['evidence']).glob('*-monitor.json'));assert len(monitors)==1
        samples=json.loads(monitors[0].read_text());assert samples and all(s['guard_passed'] for s in samples)
cache=checks/stages[-1]['accepted_cases'][0]['evidence']/'ndarray-cache.log'
assert cache.read_text().strip()=='Selective NDArray native exact/update/cache256/fallback14/cached-after-cap/abort-recovery/shape80-160/native-half PASS'
failed=checks/'attempts/ctest/case0-attempt0';status=json.loads((failed/'statuses.json').read_text())
assert len(status)==1 and status[0]['name']=='ctest' and status[0]['exit_code']==8 and status[0]['interruption'] is None
samples=json.loads((failed/'ctest-monitor.json').read_text());assert samples and all(s['guard_passed'] for s in samples)
text=(failed/'ctest.log').read_text()
assert 'NDArray prototype requires packed FP32 matrices' in text
assert '86% tests passed, 1 tests failed out of 7' in text
assert '1 - metal_wide_batch_prefill_reference (Failed)' in text
for name in ('metal_coalesced_bf16_body_reference','metal_interleaved_bf16_body_reference','metal_prefill_stream_reference',
             'metal_dirty_prefix_reset_reference','metal_mps_matmul_reference','metal_screened_head_reference'):
    lines=[line for line in text.splitlines() if 'Test #' in line and name in line];assert len(lines)==1 and 'Passed' in lines[0]
assert not (checks/'full-grid').exists()
source=(root/'source/src/backends/metal/metal_context.mm').read_text()
assert 'if((a.rows==80||a.rows==160)' in source and 'NDArray prototype requires packed FP32 matrices' in source
result=dict(verified_files=len(proof['files']),regenerated_changes=8,accepted_initial_stages=4,ctests_passed=6,ctests_failed=1,
    failed_ctest='metal_wide_batch_prefill_reference',ctest_exit_code=8,guard_interruption=False,selective_cache_primitive_pass=True,
    archive_sha256='a04efb9061130bb856a399168e23967f5ecedd6a9254632507a20508b15dbf68',scope=__doc__,full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
