"""Audit complete matched outputs, real block dispatches and original precision."""
from pathlib import Path

wrapper=Path(__file__).with_name('verify_wide_batch_vllm.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace("glob('wide-batch-vllm-20261015*')","glob('coalesced-body-vllm-20261015*')")
needle="                assert value['small_screen_seed_enabled'] and value['interleaved_screen_enabled']"
assert source.count(needle)==1
source=source.replace(needle,needle+'''
                assert value['interleaved_body_enabled'] and value['coalesced_block_enabled']
                for name in ('interleaved_body_counts','coalesced_block_counts'):
                    counts=value[name]
                    assert len(counts)==value['decode_steps_after_first']+1 and len(counts)>1
                    assert counts[0]==0 and all(v>0 for v in counts[1:]) and len(set(counts[1:]))==1
''')
exec(compile(source,__file__,'exec'))
