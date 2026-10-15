"""Durable launcher for exclusive trained common-alpha paired projections."""
from pathlib import Path
source=Path(__file__).with_name('launch_single_row_bf16_probe.py').read_text()
source=source.replace('run_single_row_bf16_probe.py','run_joined_global_alpha_probe.py')
exec(compile(source,str(Path(__file__)),'exec'))
