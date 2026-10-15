"""Preserve a terminal fresh matched block cohort with complete frozen sources."""
from pathlib import Path

wrapper=Path(__file__).with_name('archive_wide_batch_vllm.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace("glob('wide-batch-vllm-20261015*')","glob('coalesced-body-vllm-20261015*')")
source=source.replace('work/wide-batch-next-comparator-selection.json','work/coalesced-body-next-comparator-selection.json')
needle='for name in names:files[';assert source.count(needle)==1
source=source.replace(needle,"names=names+('compare_coalesced_body_vllm.py','run_coalesced_body_vllm.py','launch_coalesced_body_vllm.py','archive_coalesced_body_vllm.py','verify_coalesced_body_vllm.py')\n"+needle)
exec(compile(source,__file__,'exec'))
