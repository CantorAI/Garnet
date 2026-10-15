"""Three fresh actual GPU dtype audits followed by six sequential matched profiles."""
from pathlib import Path
import sys
wrapper=Path(__file__).with_name('run_screened_head_vllm.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('\ndestination=')],__file__,'exec'),ns)
source=ns['source'].replace('compare_screened_head_vllm.py','compare_wide_batch_vllm.py')
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
