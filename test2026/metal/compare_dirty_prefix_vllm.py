"""Fresh matched-FP32 comparison with an independently pinned reset selection."""
from pathlib import Path
import sys
wrapper=Path(__file__).with_name('compare_global_alpha_vllm.py').read_text()
ns={'__file__':__file__};exec(compile(wrapper[:wrapper.index('destination=')],__file__,'exec'),ns);source=ns['source']
source=source.replace('global-alpha-next-comparator-selection.json','dirty-prefix-next-comparator-selection.json')
needle="assert value['global_half_alpha_enabled']"
assert source.count(needle)==1
source=source.replace(needle,needle+"\n            assert value['dirty_prefix_reset_enabled']==selected['dirty_prefix_reset_enabled']\n            assert not value['vision_global_alpha_enabled'] and not value['joined_global_mlp_enabled'] and not value['single_coder_head_enabled']")
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
