"""Audit original numerical gates, native-key exactness and actual async flushes."""
from pathlib import Path

wrapper=Path(__file__).with_name('verify_hybrid_combined_graph_prefill.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace('hybrid_combined_graph_prefill_gpu.mm','async_key_graph_prefill_gpu.mm')
needle="                    assert check['native_key_bit_exact']";assert source.count(needle)==1
source=source.replace(needle,needle+"\n                    assert check['native_key_async_flushes']==int(expected_keys>0)")
source=source.replace('candidate includes native exact-key work plus graph call;', 'candidate includes native exact-key Flush, graph call and final native wait;')
exec(compile(source,__file__,'exec'))
