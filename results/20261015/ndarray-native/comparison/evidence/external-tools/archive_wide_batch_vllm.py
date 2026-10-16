"""Archive the terminal latency-bundle comparison and fresh GPU dtype evidence."""
from pathlib import Path
wrapper=Path(__file__).with_name('archive_screened_head_vllm.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace("glob('screened-head-vllm-20261015*')","glob('wide-batch-vllm-20261015*')").replace('work/screened-head-next-comparator-selection.json','work/wide-batch-next-comparator-selection.json')
needle='for name in names:files[';assert source.count(needle)==1
source=source.replace(needle,"names=names+('compare_wide_batch_vllm.py','run_wide_batch_vllm.py','launch_wide_batch_vllm.py','archive_wide_batch_vllm.py')\n"+needle)
exec(compile(source,__file__,'exec'))
