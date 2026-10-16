"""Quiet-gated durable full validation; no concurrent benchmark runtimes."""
from pathlib import Path
source=Path(__file__).with_name('launch_global_half_alpha_checks.py').read_text().replace('run_guarded_global_half_alpha_checks.py','run_guarded_fused_head_argmax_checks.py')
exec(compile(source,__file__,'exec'))
