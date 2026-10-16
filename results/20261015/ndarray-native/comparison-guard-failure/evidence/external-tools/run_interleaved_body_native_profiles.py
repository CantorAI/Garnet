"""Complete six-profile native outputs and actual body dispatch counts."""
from pathlib import Path
source=Path(__file__).with_name('run_wide_batch_prefill_profiles.py').read_text().replace('compare_wide_batch_prefill.py','compare_interleaved_body_native.py')
exec(compile(source,__file__,'exec'))
