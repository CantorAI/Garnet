"""Durable launcher for additional actual packed prefill shapes."""
from pathlib import Path
import sys
source=Path(__file__).with_name('launch_prefill_residual_norm_probe.py').read_text()
source=source.replace('run_prefill_residual_norm_probe.py','run_extra_prefill_residual_norm_probe.py')
exec(compile(source,str(Path(__file__)),'exec'))
