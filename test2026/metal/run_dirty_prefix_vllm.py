"""Prepare three fresh actual dtype audits and six sequential matched profiles."""
from pathlib import Path
import sys
wrapper=Path(__file__).with_name('run_global_alpha_vllm.py').read_text()
ns={'__file__':__file__};exec(compile(wrapper[:wrapper.index('destination=')],__file__,'exec'),ns);source=ns['source']
source=source.replace('compare_global_alpha_vllm.py','compare_dirty_prefix_vllm.py')
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
