"""Apply the same full matched audit to the separate confirmation cohort."""
from pathlib import Path
wrapper=Path(__file__).with_name('verify_wide_batch_vllm.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace("glob('wide-batch-vllm-20261015*')","glob('wide-batch-vllm-confirm-20261015*')")
exec(compile(source,__file__,'exec'))
