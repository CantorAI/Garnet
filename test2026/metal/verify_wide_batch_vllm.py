"""Audit complete matched outputs, exact values, actual GPU dtypes and warm means."""
from pathlib import Path
wrapper=Path(__file__).with_name('verify_screened_head_vllm.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace("glob('screened-head-vllm-20261015*')","glob('wide-batch-vllm-20261015*')")
source=source.replace("assert not value['prefill_stream_enabled'] and not value['dirty_prefix_reset_enabled']","assert value['prefill_stream_enabled'] and value['dirty_prefix_reset_enabled']")
source=source.replace('else [80] if batch==1 else [32,32,16]','else [80]')
needle="                assert value['screened_head_enabled'] and value['screened_head_counts']==[1]*(value['decode_steps_after_first']+1)";assert source.count(needle)==1
source=source.replace(needle,needle+'''
                assert value['small_screen_seed_enabled'] and value['interleaved_screen_enabled']
                assert value['screened_head_seed_rows']==[1024]*(value['decode_steps_after_first']+1)
                assert value['interleaved_screen_counts']==[1]*(value['decode_steps_after_first']+1)
                assert (value['prefill_profile'].get('wide_batch_prefill_dispatches',0)>0)==(model=='Qwen3.5-0.8B' and batch==2)
                if model=='Qwen3.5-0.8B':assert value['serial_vision_prefetch_enabled'] and value['vision_prefetch_dispatches']==12
''')
exec(compile(source,__file__,'exec'))
