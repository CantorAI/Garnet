"""Quiet-gated fresh retained-source build and all model validation cases."""
from pathlib import Path
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_guarded_reused_zero_source_checks.py')
exec(compile(source,__file__,'exec'))
