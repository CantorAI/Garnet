"""Fresh precision-matched comparison of the validated lossless block candidate."""
from pathlib import Path
import sys

wrapper=Path(__file__).with_name('compare_wide_batch_vllm.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('\ndestination=')],__file__,'exec'),ns);source=ns['source']
source=source.replace('work/wide-batch-next-comparator-selection.json','work/coalesced-body-next-comparator-selection.json')
needle="            assert value['small_screen_seed_enabled'] and value['interleaved_screen_enabled']"
assert source.count(needle)==1
source=source.replace(needle,needle+'''
            assert value['interleaved_body_enabled'] and value['coalesced_block_enabled']
            for name in ('interleaved_body_counts','coalesced_block_counts'):
                counts=value[name]
                assert len(counts)==value['decode_steps_after_first']+1 and len(counts)>1
                assert counts[0]==0 and all(v>0 for v in counts[1:]) and len(set(counts[1:]))==1
''')
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
