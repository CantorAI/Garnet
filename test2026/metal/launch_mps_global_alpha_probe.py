"""Durable launcher for the isolated MPS global-alpha compatibility gate."""
from pathlib import Path
source=Path(__file__).with_name('launch_single_row_bf16_probe.py').read_text()
source=source.replace('run_single_row_bf16_probe.py','run_mps_global_alpha_probe.py')
exec(compile(source,str(Path(__file__)),'exec'))
