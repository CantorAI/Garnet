"""Apply the same complete matched audit to the independent confirmation."""
from pathlib import Path

wrapper=Path(__file__).with_name('verify_coalesced_body_vllm.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace("glob('coalesced-body-vllm-20261015*')","glob('coalesced-body-vllm-confirm-20261015*')")
exec(compile(source,__file__,'exec'))
