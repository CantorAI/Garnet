"""Preserve the separate matched confirmation and all its exact launch helpers."""
from pathlib import Path

wrapper=Path(__file__).with_name('archive_coalesced_body_vllm.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace("glob('coalesced-body-vllm-20261015*')","glob('coalesced-body-vllm-confirm-20261015*')")
needle='for name in names:files[';assert source.count(needle)==1
source=source.replace(needle,"names=names+('launch_confirm_coalesced_body_vllm.py','archive_confirm_coalesced_body_vllm.py','verify_confirm_coalesced_body_vllm.py')\n"+needle)
exec(compile(source,__file__,'exec'))
