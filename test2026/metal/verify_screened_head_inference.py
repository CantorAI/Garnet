"""Audit full original gates plus trained screening and complete native outputs."""
from pathlib import Path
source=Path(__file__).with_name('verify_fused_head_argmax_inference.py').read_text()
source=source.replace('fused-head-argmax','screened-head').replace('METAL_FUSED_HEAD_ARGMAX','METAL_SCREENED_HEAD').replace('metal_fused_head_argmax_reference','metal_screened_head_reference').replace('fused_head_argmax','screened_head').replace('fused_head_session_reference PASS','screened_head_session_reference PASS').replace('actual_fused_head','actual_screened_head')
source=source.replace('len(stages)==19','len(stages)==20').replace("names={'configure'","names={'screened-real-heads','configure'").replace("dependencies['pid']==68935","dependencies['pid']==91274").replace("dependencies['completed_original_stages']==19","dependencies['completed_original_stages']==20").replace('original_validation_stages=19','original_validation_stages=19,additional_trained_head_queries=51')
source=source.replace("'private_sessions_isolated'))","'private_sessions_isolated','rebuild_and_close'))")
needle="trial=root/'logs'/native_trial";assert source.count(needle)==1
source=source.replace(needle,'''trained=json.loads((checks/'trained-heads/summary.json').read_text())
assert trained['query_checks']==51 and trained['row_checks']==78 and len(trained['profiles'])==3
assert {p['model'] for p in trained['profiles']}=={'Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'}
for p in trained['profiles']:
    asset_path='logs/screened-head-after-vision-20261015/jobs/'+p['model']+'-assets/'
    assert json.loads((root/asset_path/'manifest.json').read_text())==p['manifest']
    assert proof['omitted'][asset_path+'raw.bin']['sha256']==p['manifest']['raw_sha256']
    for q in p['manifest']['queries']:
        for suffix,key in (('.input.bin','input_sha256'),('.reference.bin','reference_sha256')):
            assert proof['omitted'][asset_path+q['name']+suffix]['sha256']==q[key]
    values=p['checks'];assert len(values)==17
    assert {v['query'] for v in values}=={q['name'] for q in p['manifest']['queries']}
    assert all(v['raw_bits_exact'] and v['retained_bits_exact'] and v['every_rejection_checked'] and v['greedy_exact'] and v['dispatches']==1 for v in values)
'''+needle)
needle="            row.update(build_policy=policy,repeat=repeat)";assert source.count(needle)==1
source=source.replace(needle,'''            assert not row['fused_head_argmax_enabled'] and not row['body_shape_enabled'] and not row['deferred_state_zero_enabled'] and not row['reused_zero_source_enabled']
            if model=='Qwen3.5-0.8B':
                assert row['frontend_policy']=='combined' and row['frontend_route']=='threaded-striped'
                assert not row['parallel_vision_context_enabled'] and row['vision_parallel_dispatches']==0
                assert all(row['rgb_sha256']==v['rgb_sha256'] for v in records)
'''+needle)
exec(compile(source,__file__,'exec'))
