"""Durable launcher for isolated packed prefill native validation."""
from pathlib import Path
source=Path(__file__).with_name('launch_residual_norm_checks.py').read_text()
source=source.replace('run_residual_norm_checks.py','run_packed_prefill_norm_checks.py')
exec(compile(source,str(Path(__file__)),'exec'))
