"""Durable launcher for exclusive full vision-matrix alpha checks."""
from pathlib import Path
source=Path(__file__).with_name('launch_single_row_bf16_probe.py').read_text().replace('run_single_row_bf16_probe.py','run_vision_global_alpha_probe.py')
exec(compile(source,str(Path(__file__)),'exec'))
