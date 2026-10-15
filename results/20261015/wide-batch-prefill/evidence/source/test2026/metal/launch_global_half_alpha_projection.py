"""Durable launcher for exclusive trained common-scale MPS projections."""
from pathlib import Path
source=Path(__file__).with_name('launch_single_row_bf16_probe.py').read_text()
source=source.replace('run_single_row_bf16_probe.py','run_global_half_alpha_projection.py')
exec(compile(source,str(Path(__file__)),'exec'))
