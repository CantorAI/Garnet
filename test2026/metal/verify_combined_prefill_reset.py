"""Full combined numerical, recovery, output, byte-count and submission audit."""
import ast
from pathlib import Path
ns={'__file__':__file__}
wrapper=Path(__file__).with_name('verify_prefill_stream_native.py').read_text()
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns)
source=ns['source'].replace('prefill-stream-checks-20261015/jobs','combined-prefill-reset-20261015/checks').replace('prefill-stream-native-20261015/jobs','combined-prefill-reset-20261015/native')
source=source.replace('assert len(stages)==19','assert len(stages)==22').replace('numerical_stages=19','numerical_stages=22')
source=source.replace("not r['dirty_prefix_reset_enabled']", "r['dirty_prefix_reset_enabled']==candidate")
source=source.replace("'METAL_PREFILL_STREAM' in meta['options']", "'METAL_PREFILL_STREAM' in meta['options'] and 'METAL_DIRTY_PREFIX_RESET' in meta['options']")
tree=ast.parse(Path(__file__).with_name('verify_dirty_prefix_reset_native.py').read_text())
blocks=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and "configfile=base/'models'" in n.value and "r['dirty_prefix_reset_bytes']" in n.value]
assert len(blocks)==1
needle="assert r['vision_global_alpha_dispatches']==0 and r['decode_partial_submissions']==0"
assert source.count(needle)==1
source=source.replace(needle,needle+'\n  '+blocks[0])
needle="meta=json.loads((checks/'metadata.json').read_text())"
assert source.count(needle)==1
recovery='''ctest=(checks/'CTest-LastTest.log').read_text()
assert all(name in ctest for name in ('metal_prefill_stream_reference','metal_dirty_prefix_reset_reference'))
assert 'Dirty prefix reset complete CPU/GPU positive zero, bounds, uninitialized and abort PASS' in ctest
assert {v['name'] for v in stages[19:]}=={m+'-session-recovery' for m in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B')}
for stage in stages[19:]:
 accepted=stage['accepted_cases'];assert len(accepted)==1
 where=checks/accepted[0]['evidence']
 text=(where/(stage['name']+'.log')).read_text()
 results=[json.loads(line.split(' ',2)[2]) for line in text.splitlines() if line.startswith('prefix_reset_recovery PASS ')]
 assert len(results)==2 and {v['batch'] for v in results}=={1,2}
 assert all(all(v[k] for k in ('partial_clear_verified','unknown_writer_full_fallback','nonfinite_request_poisoned','full_recovery_verified','private_sessions_isolated')) for v in results)
'''
source=source.replace(needle,recovery+needle)
source=source.replace('Ordered prefill submissions versus global alpha; no fresh vLLM comparison','Combined bounded reset and ordered prefill versus global alpha; no fresh vLLM comparison')
source=source.replace('all_actual_prefill_partial_submissions_checked=True','all_actual_prefill_partial_submissions_checked=True,all_actual_dirty_prefix_reset_bytes_checked=True,all_three_model_recovery_suites_checked=True')
if '--confirm' in __import__('sys').argv[3:]:
    source=source.replace('combined-prefill-reset-20261015/native','combined-prefill-reset-confirm-20261015/jobs')
exec(compile(source,__file__,'exec'))
