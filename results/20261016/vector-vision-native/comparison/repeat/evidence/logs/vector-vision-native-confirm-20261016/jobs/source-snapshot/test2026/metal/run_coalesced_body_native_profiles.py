"""All six complete native profiles, with actual lossless block dispatch counts."""
from pathlib import Path
source=Path(__file__).with_name('run_wide_batch_prefill_profiles.py').read_text()
source=source.replace('compare_wide_batch_prefill.py','compare_coalesced_body_native.py')
exec(compile(source,__file__,'exec'))
